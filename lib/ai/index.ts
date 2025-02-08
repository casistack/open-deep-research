import { openai } from '@ai-sdk/openai';
import { wrapLanguageModel, customProvider, type LanguageModelV1, type LanguageModelV1CallOptions, type LanguageModelV1StreamPart } from 'ai';
import { openrouter } from '@openrouter/ai-sdk-provider';
import { togetherai } from '@ai-sdk/togetherai';
import { deepseek } from '@ai-sdk/deepseek';

import { customMiddleware } from "./custom-middleware";

// Default model to use if none specified
const DEFAULT_MODEL = 'gpt-3.5-turbo';

// Create LiteLLM provider instance with model ID
const createLiteLLMProvider = (baseUrl: string, modelId?: string) => {
  const apiKey = process.env.LITELLM_API_KEY;
  return customProvider({
    languageModels: {
      default: new LiteLLMModel(baseUrl, modelId, apiKey)
    }
  });
};

// LiteLLM language model implementation
class LiteLLMModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1';
  readonly provider = 'litellm';
  readonly modelId: string;
  readonly defaultObjectGenerationMode = 'json';

  private readonly headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  constructor(
    private readonly baseUrl: string,
    modelId?: string,
    apiKey?: string
  ) {
    this.modelId = modelId || DEFAULT_MODEL;
    if (apiKey) {
      this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
  }

  async doStream(options: LanguageModelV1CallOptions) {
    // Extract messages from the prompt
    const messages = Array.isArray(options.prompt)
      ? options.prompt
      : [{ role: 'user', content: String(options.prompt) }];

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        stream: true,
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        model: this.modelId
      })
    });

    if (!response.ok) {
      throw new Error(`LiteLLM request failed: ${response.statusText}`);
    }

    // Create a readable stream that transforms the response into LanguageModelV1StreamPart
    const decoder = new TextDecoder();
    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      async start(controller) {
        const reader = response.body!.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const textDelta = decoder.decode(value);
            controller.enqueue({ type: 'text-delta', textDelta });
          }
        } finally {
          reader.releaseLock();
          controller.close();
        }
      }
    });

    return {
      stream,
      rawCall: {
        rawPrompt: messages,
        rawSettings: options
      }
    };
  }

  async doGenerate(options: LanguageModelV1CallOptions) {
    // Extract messages from the prompt
    const messages = Array.isArray(options.prompt)
      ? options.prompt
      : [{ role: 'user', content: String(options.prompt) }];

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        stream: false,
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        model: this.modelId
      })
    });

    if (!response.ok) {
      throw new Error(`LiteLLM request failed: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      text: data.choices[0]?.message?.content ?? '',
      finishReason: data.choices[0]?.finish_reason ?? 'stop',
      usage: {
        promptTokens: data.usage?.prompt_tokens ?? 0,
        completionTokens: data.usage?.completion_tokens ?? 0,
        totalTokens: data.usage?.total_tokens ?? 0
      },
      rawCall: {
        rawPrompt: messages,
        rawSettings: options
      }
    };
  }
}

// Type definition for valid reasoning models used for research and structured outputs
type ReasoningModel = typeof VALID_REASONING_MODELS[number];

// Valid reasoning models that can be used for research analysis and structured outputs
const VALID_REASONING_MODELS = [
  'o1', 'o1-mini', 'o3-mini',
  'deepseek-ai/DeepSeek-R1',
  'deepseek-reasoner',
  'gpt-4o',
  'deepseek-r1:32b',
] as const;

// Models that support JSON structured output
const JSON_SUPPORTED_MODELS = ['gpt-4o', 'gpt-4o-mini', 'o1', 'o3-mini', 'deepseek-r1:32b'] as const;

// Helper to check if model supports JSON
export const supportsJsonOutput = (modelId: string) =>
  JSON_SUPPORTED_MODELS.includes(modelId as typeof JSON_SUPPORTED_MODELS[number]);

// Get reasoning model from env, with JSON support info
const REASONING_MODEL = process.env.REASONING_MODEL || 'o1-mini';
const BYPASS_JSON_VALIDATION = process.env.BYPASS_JSON_VALIDATION === 'true';

// Helper to get the reasoning model based on user's selected model
function getReasoningModel(modelId: string) {
  // If already using a valid reasoning model, keep using it
  if (VALID_REASONING_MODELS.includes(modelId as ReasoningModel)) {
    return modelId;
  }

  const configuredModel = REASONING_MODEL;

  if (!VALID_REASONING_MODELS.includes(configuredModel as ReasoningModel)) {
    const fallback = 'o1-mini';
    console.warn(`Invalid REASONING_MODEL "${configuredModel}", falling back to ${fallback}`);
    return fallback;
  }

  // Warn if trying to use JSON with unsupported model
  if (!BYPASS_JSON_VALIDATION && !supportsJsonOutput(configuredModel)) {
    console.warn(`Warning: Model ${configuredModel} does not support JSON schema. Set BYPASS_JSON_VALIDATION=true to override`);
  }

  return configuredModel;
}

export const customModel = (apiIdentifier: string, forReasoning: boolean = false) => {
  // Check which API services are available
  const hasLiteLLMServer = process.env.LITELLM_API_URL && process.env.LITELLM_API_URL !== "";
  const hasOpenRouterKey = process.env.OPENROUTER_API_KEY && process.env.OPENROUTER_API_KEY !== "****";

  // If it's for reasoning, get the appropriate reasoning model
  const modelId = forReasoning ? getReasoningModel(apiIdentifier) : apiIdentifier;

  // Try LiteLLM first if available
  if (hasLiteLLMServer) {
    const liteLLMProvider = createLiteLLMProvider(process.env.LITELLM_API_URL!, modelId);
    return wrapLanguageModel({
      model: liteLLMProvider.languageModel('default'),
      middleware: customMiddleware,
    });
  }

  // Fall back to OpenRouter if available
  if (hasOpenRouterKey) {
    return wrapLanguageModel({
      model: openrouter(modelId),
      middleware: customMiddleware,
    });
  }

  // Fall back to other providers based on model
  const model = modelId === 'deepseek-ai/DeepSeek-R1'
    ? togetherai(modelId)
    : modelId === 'deepseek-reasoner'
    ? deepseek(modelId)
    : openai(modelId);

  return wrapLanguageModel({
    model,
    middleware: customMiddleware,
  });
};
