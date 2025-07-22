import type {
  CoreMessage,
  GenerateTextResult as OriginalGenerateTextResult,
  GenerateObjectResult as OriginalGenerateObjectResult,
  StreamTextResult as OriginalStreamTextResult,
  StreamObjectResult as OriginalStreamObjectResult,
  TelemetrySettings,
  UIMessage,
  Tool,
  generateText,
  ToolSet,
  generateObject,
  streamText,
  streamObject,
  DeepPartial,
  StreamObjectOnFinishCallback as OriginalStreamObjectOnFinishCallback,
  StreamTextOnFinishCallback as OriginalStreamTextOnFinishCallback,
  StreamTextOnStepFinishCallback as OriginalStreamTextOnStepFinishCallback,
  GenerateTextOnStepFinishCallback as OriginalGenerateTextOnStepFinishCallback,
} from 'ai';
import type { JSONSchema7 } from 'json-schema';
import type { ZodSchema, z } from 'zod';
import type { RuntimeContext } from '../../runtime-context';

export type inferOutput<Output extends ZodSchema | JSONSchema7 | undefined = undefined> = Output extends ZodSchema
  ? z.infer<Output>
  : Output extends JSONSchema7
    ? unknown
    : undefined;

export type { ToolSet } from 'ai';

type MastraCustomLLMOptions = {
  tools?: Record<string, Tool>;
  telemetry?: TelemetrySettings;
  threadId?: string;
  resourceId?: string;
  runtimeContext: RuntimeContext;
  runId?: string;
};
type MastraCustomLLMOptionsKeys = keyof MastraCustomLLMOptions;

export type OriginalStreamTextOnFinishEventArg<Tools extends ToolSet> = Parameters<
  OriginalStreamTextOnFinishCallback<Tools>
>[0];
export type OriginalStreamObjectOnFinishEventArg<RESULT> = Parameters<OriginalStreamObjectOnFinishCallback<RESULT>>[0];

export type StreamTextOnFinishCallback<Tools extends ToolSet> = (
  event: OriginalStreamTextOnFinishEventArg<Tools> & { runId: string },
) => Promise<void> | void;
export type StreamObjectOnFinishCallback<RESULT> = (
  event: OriginalStreamObjectOnFinishEventArg<RESULT> & { runId: string },
) => Promise<void> | void;

export type GenerateTextOnStepFinishCallback<Tools extends ToolSet> = (
  event: Parameters<OriginalGenerateTextOnStepFinishCallback<Tools>>[0] & { runId: string },
) => Promise<void> | void;

export type StreamTextOnStepFinishCallback<Tools extends ToolSet> = (
  event: Parameters<OriginalStreamTextOnStepFinishCallback<Tools>>[0] & { runId: string },
) => Promise<void> | void;

// #region generateText
export type OriginalGenerateTextOptions<
  TOOLS extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
> = Parameters<typeof generateText<TOOLS, inferOutput<Output>, DeepPartial<inferOutput<Output>>>>[0];
type GenerateTextOptions<Tools extends ToolSet, Output extends ZodSchema | JSONSchema7 | undefined = undefined> = Omit<
  OriginalGenerateTextOptions<Tools, Output>,
  MastraCustomLLMOptionsKeys | 'model' | 'onStepFinish'
> &
  MastraCustomLLMOptions & {
    onStepFinish?: GenerateTextOnStepFinishCallback<inferOutput<Output>>;
    experimental_output?: Output;
  };

export type GenerateTextWithMessagesArgs<
  Tools extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
> = {
  messages: UIMessage[] | CoreMessage[];
  output?: never;
} & GenerateTextOptions<Tools, Output>;

export type GenerateTextResult<
  Tools extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
> = Omit<OriginalGenerateTextResult<Tools, inferOutput<Output>>, 'experimental_output'> & {
  object?: Output extends undefined ? never : inferOutput<Output>;
};

export type OriginalGenerateObjectOptions<Output extends ZodSchema | JSONSchema7 | undefined = undefined> =
  | Parameters<typeof generateObject<inferOutput<Output>>>[0]
  | (Parameters<typeof generateObject<inferOutput<Output>>>[0] & { output: 'array' })
  | (Parameters<typeof generateObject<string>>[0] & { output: 'enum' })
  | (Parameters<typeof generateObject>[0] & { output: 'no-schema' });

type GenerateObjectOptions<Output extends ZodSchema | JSONSchema7 | undefined = undefined> = Omit<
  OriginalGenerateObjectOptions<Output>,
  MastraCustomLLMOptionsKeys | 'model' | 'output'
> &
  MastraCustomLLMOptions;

export type GenerateObjectWithMessagesArgs<Output extends ZodSchema | JSONSchema7> = {
  messages: UIMessage[] | CoreMessage[];
  structuredOutput: Output;
  output?: never;
} & GenerateObjectOptions<Output>;

export type GenerateObjectResult<Output extends ZodSchema | JSONSchema7 | undefined = undefined> =
  OriginalGenerateObjectResult<inferOutput<Output>> & {
    readonly reasoning?: never;
  };

export type GenerateReturn<
  Tools extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
  StructuredOutput extends ZodSchema | JSONSchema7 | undefined = undefined,
> = GenerateTextResult<Tools, StructuredOutput> | GenerateObjectResult<Output>;
// #endregion

// #region streamText
export type OriginalStreamTextOptions<
  TOOLS extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
> = Parameters<typeof streamText<TOOLS, inferOutput<Output>, DeepPartial<inferOutput<Output>>>>[0];
type StreamTextOptions<Tools extends ToolSet, Output extends ZodSchema | JSONSchema7 | undefined = undefined> = Omit<
  OriginalStreamTextOptions<Tools, Output>,
  MastraCustomLLMOptionsKeys | 'model' | 'onStepFinish' | 'onFinish'
> &
  MastraCustomLLMOptions & {
    onStepFinish?: StreamTextOnStepFinishCallback<inferOutput<Output>>;
    onFinish?: StreamTextOnFinishCallback<inferOutput<Output>>;
    experimental_output?: Output;
  };

export type StreamTextWithMessagesArgs<
  Tools extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
> = {
  messages: UIMessage[] | CoreMessage[];
  output?: never;
} & StreamTextOptions<Tools, Output>;

export type StreamTextResult<
  Tools extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
> = Omit<OriginalStreamTextResult<Tools, DeepPartial<inferOutput<Output>>>, 'experimental_output'> & {
  object?: inferOutput<Output>;
};

export type OriginalStreamObjectOptions<Output extends ZodSchema | JSONSchema7> =
  | Parameters<typeof streamObject<inferOutput<Output>>>[0]
  | (Parameters<typeof streamObject<inferOutput<Output>>>[0] & { output: 'array' })
  | (Parameters<typeof streamObject<string>>[0] & { output: 'enum' })
  | (Parameters<typeof streamObject>[0] & { output: 'no-schema' });

type StreamObjectOptions<Output extends ZodSchema | JSONSchema7> = Omit<
  OriginalStreamObjectOptions<Output>,
  MastraCustomLLMOptionsKeys | 'model' | 'output' | 'onFinish'
> &
  MastraCustomLLMOptions & {
    onFinish?: StreamObjectOnFinishCallback<inferOutput<Output>>;
  };

export type StreamObjectWithMessagesArgs<Output extends ZodSchema | JSONSchema7> = {
  messages: UIMessage[] | CoreMessage[];
  structuredOutput: Output;
  output?: never;
} & StreamObjectOptions<Output>;

export type StreamObjectResult<Output extends ZodSchema | JSONSchema7> = OriginalStreamObjectResult<
  DeepPartial<inferOutput<Output>>,
  inferOutput<Output>,
  any
>;

export type StreamReturn<
  Tools extends ToolSet,
  Output extends ZodSchema | JSONSchema7 | undefined = undefined,
  StructuredOutput extends ZodSchema | JSONSchema7 | undefined = undefined,
> = StreamTextResult<Tools, StructuredOutput> | StreamObjectResult<NonNullable<Output>>;
// #endregion
