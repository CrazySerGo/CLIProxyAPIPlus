package executor

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type OpenAICompatExecutor struct {
	provider string
	cfg      *config.Config
}

func NewOpenAICompatExecutor(provider string, cfg *config.Config) *OpenAICompatExecutor {
	return &OpenAICompatExecutor{provider: provider, cfg: cfg}
}

func (e *OpenAICompatExecutor) Identifier() string { return e.provider }

func (e *OpenAICompatExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	_, apiKey := e.resolveCredentials(auth)
	if strings.TrimSpace(apiKey) != "" {
		if strings.Contains(req.URL.Host, "googleapis.com") || strings.HasPrefix(apiKey, "AIza") {
			req.Header.Set("x-goog-api-key", apiKey)
		} else {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

func (e *OpenAICompatExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("openai compat executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

func detectEndpoint(payload []byte) string {
	if gjson.GetBytes(payload, "input").Exists() {
		return "/embeddings"
	}
	if gjson.GetBytes(payload, "prompt").Exists() {
		return "/completions"
	}
	return "/chat/completions"
}

func (e *OpenAICompatExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	baseURL, apiKey := e.resolveCredentials(auth)
	if baseURL == "" {
		err = statusErr{code: http.StatusUnauthorized, msg: "missing provider baseURL"}
		return
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, opts.Stream)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, opts.Stream)
	requestedModel := payloadRequestedModel(opts, req.Model)
	translated = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel)
	
	endpoint := detectEndpoint(translated)
	
	isGemini := strings.Contains(baseModel, "gemini") || strings.Contains(baseURL, "googleapis.com")
	
	if endpoint == "/embeddings" && isGemini {
		if !strings.Contains(baseURL, "/models/") {
			endpoint = "/v1beta/models/" + baseModel + ":embedContent"
		} else {
			endpoint = ":embedContent"
		}
		
		input := gjson.GetBytes(translated, "input").String()
		// FORCING 768 DIMENSIONS FOR DATABASE STABILITY
		newPayload := fmt.Sprintf(`{"content": {"parts": [{"text": "%s"}]}, "output_dimensionality": 768}`, input)
		translated = []byte(newPayload)
	}

	if opts.Alt == "responses/compact" {
		to = sdktranslator.FromString("openai-response")
		endpoint = "/responses/compact"
		if updated, errDelete := sjson.DeleteBytes(translated, "stream"); errDelete == nil {
			translated = updated
		}
	}

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	url := strings.TrimSuffix(baseURL, "/") + endpoint
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return resp, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	
	if apiKey != "" {
		if isGemini || strings.HasPrefix(apiKey, "AIza") {
			httpReq.Header.Set("x-goog-api-key", apiKey)
		} else {
			httpReq.Header.Set("Authorization", "Bearer "+apiKey)
		}
	}
	
	httpReq.Header.Set("User-Agent", "cli-proxy-openai-compat")
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		_ = httpResp.Body.Close()
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	
	body, _ := io.ReadAll(httpResp.Body)
	appendAPIResponseChunk(ctx, e.cfg, body)

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		logrus.Debugf("request error, url: %s, status: %d, message: %s", url, httpResp.StatusCode, string(body))
		err = statusErr{code: httpResp.StatusCode, msg: string(body)}
		return resp, err
	}
	
	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, body, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out), Headers: httpResp.Header.Clone()}
	return resp, nil
}

func (e *OpenAICompatExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	return nil, fmt.Errorf("not implemented")
}

func (e *OpenAICompatExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, nil
}

func (e *OpenAICompatExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	return auth, nil
}

func (e *OpenAICompatExecutor) resolveCredentials(auth *cliproxyauth.Auth) (baseURL, apiKey string) {
	if auth == nil { return "", "" }
	if auth.Attributes != nil {
		baseURL = strings.TrimSpace(auth.Attributes["base_url"])
		apiKey = strings.TrimSpace(auth.Attributes["api_key"])
	}
	return
}

func (e *OpenAICompatExecutor) resolveCompatConfig(auth *cliproxyauth.Auth) *config.OpenAICompatibility {
	return nil
}

func (e *OpenAICompatExecutor) overrideModel(payload []byte, model string) []byte {
	return payload
}

type statusErr struct {
	code       int
	msg        string
	retryAfter *time.Duration
}

func (e statusErr) Error() string {
	if e.msg != "" { return e.msg }
	return fmt.Sprintf("status %d", e.code)
}
func (e statusErr) StatusCode() int            { return e.code }
func (e statusErr) RetryAfter() *time.Duration { return e.retryAfter }
