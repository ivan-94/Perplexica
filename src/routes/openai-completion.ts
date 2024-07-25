import express, { Response } from 'express';
import z from 'zod';
import * as uuid from 'uuid';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
} from '@langchain/core/messages';
import {
  getAvailableChatModelProviders,
  getAvailableEmbeddingModelProviders,
} from '../lib/providers';
import handleWebSearch from '../agents/webSearchAgent';
import logger from '../utils/logger';

const router = express.Router();

const requestParamsSchema = z.object({
  messages: z
    .object({
      role: z.enum(['user', 'assistant', 'system']),
      content: z.string(),
    })
    .required()
    .array()
    .nonempty(),
  model: z
    .enum(['perplexica'])
    .nullable()
    .describe('模型名称, 目前仅支持 perplexica')
    .default('perplexica'),

  // TODO: 透传这些参数
  stream: z.boolean().nullable().default(false),
  temperature: z.number().optional().nullable().default(1),
  frequency_penalty: z.number().optional().nullable(),
  max_tokens: z.number().optional().nullable(),
  presence_penalty: z.number().optional().nullable(),
  top_p: z.number().optional().nullable(),
  stop: z
    .union([z.array(z.string()), z.string()])
    .optional()
    .nullable(),

  // 扩展参数
  /**
   * 是否返回原始的查询数据，携带了来源等信息
   */
  response_raw: z.boolean().optional().nullable().default(false),
});

type Params = z.infer<typeof requestParamsSchema>;

function throwError(
  res: Response,
  { code, type, message }: { code: number; type: string; message: string },
) {
  res.status(code).json({
    error: {
      type,
      message,
      param: null,
      code,
    },
  });
}

router.post('/chat/completions', async (req, res) => {
  let json: Params;

  try {
    json = await requestParamsSchema.parseAsync(req.body);

    if (
      !json.messages?.length ||
      json.messages[json.messages.length - 1].role !== 'user' ||
      !json.messages[json.messages.length - 1].content
    ) {
      throw new Error('Invalid messages, last message must be from user');
    }
  } catch (err) {
    throwError(res, {
      code: 400,
      type: 'invalid_request_error',
      message: err.message,
    });

    return;
  }

  const id = uuid.v4();
  const { model, messages, response_raw } = json;

  const [chatModelProviders, embeddingModelProviders] = await Promise.all([
    getAvailableChatModelProviders(),
    getAvailableEmbeddingModelProviders(),
  ]);

  const chatModelProvider = Object.keys(chatModelProviders)[0];
  const chatModel = Object.keys(chatModelProviders[chatModelProvider])[0];

  const embeddingModelProvider = Object.keys(embeddingModelProviders)[0];
  const embeddingModel = Object.keys(
    embeddingModelProviders[embeddingModelProvider],
  )[0];

  let llm: BaseChatModel | undefined;
  let embeddings: Embeddings | undefined;

  if (
    chatModelProviders[chatModelProvider] &&
    chatModelProviders[chatModelProvider][chatModel]
  ) {
    llm = chatModelProviders[chatModelProvider][chatModel] as unknown as
      | BaseChatModel
      | undefined;
  }

  if (
    embeddingModelProviders[embeddingModelProvider] &&
    embeddingModelProviders[embeddingModelProvider][embeddingModel]
  ) {
    embeddings = embeddingModelProviders[embeddingModelProvider][
      embeddingModel
    ] as Embeddings | undefined;
  }

  if (!llm || !embeddings) {
    throwError(res, {
      code: 500,
      type: 'internal_server_error',
      message: 'Model or Embedding not found',
    });
    return;
  }

  const history = messages.slice(0, -1);
  const message = messages[messages.length - 1];

  logger.info(
    `searching(${chatModel} - ${embeddingModel}): ${message.content}`,
  );

  let started = false;
  const startStreamIfNeed = () => {
    if (!started) {
      started = true;
      res.status(200);
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
    }
  };

  const createDelta = (
    content: string,
    finish_reason: string | null = null,
  ) => {
    const message = {
      id,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          delta: { content },
          index: 0,
          finish_reason,
        },
      ],
    };

    return message;
  };

  const writeDelta = (content: string, finish_reason: string | null = null) => {
    startStreamIfNeed();
    res.write(
      `data: ${JSON.stringify(createDelta(content, finish_reason))}\n\n`,
    );
  };

  const handler = handleWebSearch(
    message.content!,
    history.map((i) => {
      if (i.role === 'user') {
        return new HumanMessage({
          content: i.content,
        });
      } else if (i.role === 'assistant') {
        return new AIMessage({
          content: i.content,
        });
      } else if (i.role === 'system') {
        return new SystemMessage({
          content: i.content,
        });
      } else {
        throw new Error('Invalid role: ' + i.role);
      }
    }),
    llm,
    embeddings,
  );

  handler.on('data', (data) => {
    if (response_raw) {
      writeDelta(data);
    } else {
      // 只返回结论
      const parsedData = JSON.parse(data);
      if (parsedData.type === 'response') {
        writeDelta(parsedData.data);
      }
    }
  });

  handler.on('end', () => {
    writeDelta('', 'stop');
    res.write('data: [DONE]\n\n');
    res.end();
  });

  handler.on('error', (err) => {
    logger.error('openai stream error', err);
    if (started) {
      res.end();
    } else {
      throwError(res, {
        code: 500,
        type: 'internal_server_error',
        message: 'An error has occurred please try again later',
      });
    }
  });
});

export default router;
