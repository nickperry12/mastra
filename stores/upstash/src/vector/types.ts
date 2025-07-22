import type { SparseVector as UpstashSparseVector } from '../../../../packages/core/src/vector/types';
import type { UpstashVectorFilter } from './filter';
import type {
  QueryVectorParams,
  UpsertVectorParams,
} from '@mastra/core/vector';
import type {
  FusionAlgorithm,
  QueryMode,
} from '@upstash/vector';

export interface UpstashVectorPoint {
  id: string;
  vector: number[];
  sparseVector?: UpstashSparseVector;
  metadata?: Record<string, any>;
}

export interface UpstashUpsertVectorParams extends UpsertVectorParams {
  sparseVectors?: UpstashSparseVector[];
}

export interface UpstashQueryVectorParams extends QueryVectorParams<UpstashVectorFilter> {
  sparseVector?: UpstashSparseVector;
  fusionAlgorithm?: FusionAlgorithm.RRF | FusionAlgorithm.DBSF;
  queryMode?: QueryMode;
}