import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import {
  getOpenaiApiKey,
  getOpenaiApiEndpoint,
  getOpenaiModel,
  getOpenaiEmbeddingsModel,
} from '../../config';
import logger from '../../utils/logger';

export const loadOpenAIChatModels = async () => {
  const openAIApiKey = getOpenaiApiKey();
  const apiEndpoint = getOpenaiApiEndpoint();
  const model = getOpenaiModel();

  if (!openAIApiKey) return {};

  if (apiEndpoint) {
    logger.info(`OpenAI 使用自定义 API 端点： ${apiEndpoint}`);
  }

  try {
    const baseConfig = {
      openAIApiKey,
      configuration: {
        baseURL: apiEndpoint,
      },
    };

    if (model) {
      logger.info(`OpenAI 使用自定义模型： ${model}`);

      return {
        [model]: new ChatOpenAI({
          modelName: model,
          temperature: 0.7,
          ...baseConfig,
        }),
      };
    }

    const chatModels = {
      'GPT-3.5 turbo': new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        ...baseConfig,
      }),
      'GPT-4': new ChatOpenAI({
        modelName: 'gpt-4',
        temperature: 0.7,
        ...baseConfig,
      }),
      'GPT-4 turbo': new ChatOpenAI({
        modelName: 'gpt-4-turbo',
        temperature: 0.7,
        ...baseConfig,
      }),
      'GPT-4 omni': new ChatOpenAI({
        modelName: 'gpt-4o',
        temperature: 0.7,
        ...baseConfig,
      }),
      'GPT-4 omni mini': new ChatOpenAI({
        modelName: 'gpt-4o-mini',
        temperature: 0.7,
        ...baseConfig,
      }),
    };

    return chatModels;
  } catch (err) {
    logger.error(`Error loading OpenAI models: ${err}`);
    return {};
  }
};

export const loadOpenAIEmbeddingsModels = async () => {
  const openAIApiKey = getOpenaiApiKey();
  const apiEndpoint = getOpenaiApiEndpoint();
  const model = getOpenaiEmbeddingsModel();

  if (!openAIApiKey) return {};

  if (apiEndpoint) {
    logger.info(`OpenAI 使用自定义 API 端点： ${apiEndpoint}`);
  }

  try {
    const baseConfig = {
      openAIApiKey,
      configuration: {
        baseURL: apiEndpoint,
      },
    };

    if (model) {
      logger.info(`OpenAI 使用自定义嵌入模型： ${model}`);

      return {
        [model]: new OpenAIEmbeddings({
          modelName: model,
          ...baseConfig,
        }),
      };
    }

    const embeddingModels = {
      'Text embedding 3 small': new OpenAIEmbeddings({
        modelName: 'text-embedding-3-small',
        ...baseConfig,
      }),
      'Text embedding 3 large': new OpenAIEmbeddings({
        modelName: 'text-embedding-3-large',
        ...baseConfig,
      }),
    };

    return embeddingModels;
  } catch (err) {
    logger.error(`Error loading OpenAI embeddings model: ${err}`);
    return {};
  }
};
