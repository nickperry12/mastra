import { ReadableStream } from 'stream/web';
import type { Run } from '../workflows';

export type ChunkType = {
  type: string;
  runId: string;
  from: string;
  payload: Record<string, any>;
};

export class MastraWorkflowStream extends ReadableStream<ChunkType> {
  #usageCount = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };
  #streamPromise: {
    promise: Promise<void>;
    resolve: (value: void) => void;
    reject: (reason?: any) => void;
  };
  #run: Run;

  constructor({
    createStream,
    run,
  }: {
    createStream: (writer: WritableStream<ChunkType>) => Promise<ReadableStream<any>> | ReadableStream<any>;
    run: Run;
  }) {
    const deferredPromise = {
      promise: null,
      resolve: null,
      reject: null,
    } as unknown as {
      promise: Promise<void>;
      resolve: (value: void) => void;
      reject: (reason?: any) => void;
    };
    deferredPromise.promise = new Promise((resolve, reject) => {
      deferredPromise.resolve = resolve;
      deferredPromise.reject = reject;
    });

    const updateUsageCount = (usage: {
      promptTokens?: `${number}` | number;
      completionTokens?: `${number}` | number;
      totalTokens?: `${number}` | number;
    }) => {
      this.#usageCount.promptTokens += parseInt(usage.promptTokens?.toString() ?? '0', 10);
      this.#usageCount.completionTokens += parseInt(usage.completionTokens?.toString() ?? '0', 10);
      this.#usageCount.totalTokens += parseInt(usage.totalTokens?.toString() ?? '0', 10);
    };

    super({
      start: async controller => {
        const writer = new WritableStream<ChunkType>({
          write: chunk => {
            if (
              (chunk.type === 'step-output' &&
                chunk.payload?.output?.from === 'AGENT' &&
                chunk.payload?.output?.type === 'finish') ||
              (chunk.type === 'step-output' &&
                chunk.payload?.output?.from === 'WORKFLOW' &&
                chunk.payload?.output?.type === 'finish')
            ) {
              const finishPayload = chunk.payload?.output.payload;
              updateUsageCount(finishPayload.usage);
            }

            controller.enqueue(chunk);
          },
        });

        controller.enqueue({
          type: 'start',
          runId: run.runId,
          from: 'WORKFLOW',
          payload: {},
        });

        const stream = await createStream(writer);

        for await (const chunk of stream) {
          // update the usage count
          if (
            (chunk.type === 'step-output' &&
              chunk.payload?.output?.from === 'AGENT' &&
              chunk.payload?.output?.type === 'finish') ||
            (chunk.type === 'step-output' &&
              chunk.payload?.output?.from === 'WORKFLOW' &&
              chunk.payload?.output?.type === 'finish')
          ) {
            const finishPayload = chunk.payload?.output.payload;
            updateUsageCount(finishPayload.usage);
          }

          controller.enqueue(chunk);
        }

        controller.enqueue({
          type: 'finish',
          runId: run.runId,
          from: 'WORKFLOW',
          payload: {
            totalUsage: this.#usageCount,
          },
        });

        stream;

        controller.close();
        deferredPromise.resolve();
      },
    });

    this.#run = run;
    this.#streamPromise = deferredPromise;
  }

  get status() {
    return this.#streamPromise.promise.then(() => this.#run._getExecutionResults()).then(res => res!.status);
  }

  get result() {
    return this.#streamPromise.promise.then(() => this.#run._getExecutionResults());
  }

  get usage() {
    return this.#streamPromise.promise.then(() => this.#usageCount);
  }
}
