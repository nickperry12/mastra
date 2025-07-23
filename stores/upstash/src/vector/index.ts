import { MastraError, ErrorDomain, ErrorCategory } from '@mastra/core/error';
import { MastraVector } from '@mastra/core/vector';
import type {
  CreateIndexParams,
  DeleteIndexParams,
  DeleteVectorParams,
  DescribeIndexParams,
  IndexStats,
  QueryResult,
} from '@mastra/core/vector';
import { Index } from '@upstash/vector';
import { UpstashFilterTranslator } from './filter';
import type { UpstashVectorFilter } from './filter';
import type { SparseVector as UpstashSparseVector } from '../../../../packages/core/src/vector/types';
import type {
  UpstashVectorPoint,
  UpstashUpsertVectorParams,
  UpstashQueryVectorParams,
  UpstashUpdateVectorParams
} from './types';


export class UpstashVector extends MastraVector<UpstashVectorFilter> {
  private client: Index;

  /**
   * Creates a new UpstashVector instance.
   * @param {object} params - The parameters for the UpstashVector.
   * @param {string} params.url - The URL of the Upstash vector index.
   * @param {string} params.token - The token for the Upstash vector index.
   */
  constructor({ url, token }: { url: string; token: string }) {
    super();
    this.client = new Index({
      url,
      token,
    });
  }

  /**
   * Validates sparse vectors to ensure indices and values arrays match in length
   */
  private validateSparseVectors(sparseVectors: UpstashSparseVector[]): void {
    for (let i = 0; i < sparseVectors.length; i++) {
      const sparse = sparseVectors[i];
      if (sparse && sparse.indices.length !== sparse.values.length) {
        throw new MastraError({
          id: 'STORAGE_UPSTASH_VECTOR_SPARSE_VECTOR_MISMATCH',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.USER,
          text: `Sparse vector at index ${i} has mismatched indices and values lengths`,
          details: { index: i, indicesLength: sparse.indices.length, valuesLength: sparse.values.length },
        });
      }
    }
  }

  /**
   * Creates a vector point for upsert operation
   */
  private createVectorPoint(
    vector: number[],
    id: string,
    metadata?: Record<string, any>,
    sparseVector?: UpstashSparseVector,
  ): UpstashVectorPoint {
    const point: UpstashVectorPoint = {
      id,
      vector,
    };

    if (metadata) {
      point.metadata = metadata;
    }

    // Only include sparseVector if it exists and has data
    if (sparseVector && sparseVector.indices.length > 0) {
      point.sparseVector = sparseVector;
    }

    return point;
  }

  /**
   * Upserts vectors into the index.
   * @param {UpsertVectorParams} params - The parameters for the upsert operation.
   * @returns {Promise<string[]>} A promise that resolves to the IDs of the upserted vectors.
   */
  async upsert({ indexName: namespace, vectors, sparseVectors, metadata, ids }: UpstashUpsertVectorParams): Promise<string[]> {
    const generatedIds = ids || vectors.map(() => crypto.randomUUID());

    // Validate sparse vectors if provided
    if (sparseVectors) {
      this.validateSparseVectors(sparseVectors);
    }

    const points: UpstashVectorPoint[] = vectors.map((vector, index) =>
      this.createVectorPoint(
        vector,
        generatedIds[index]!,
        metadata?.[index],
        sparseVectors?.[index],
      ),
    );

    try {
      await this.client.upsert(points, {
        namespace,
      });
      return generatedIds;
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_UPSERT_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace, vectorCount: vectors.length },
        },
        error,
      );
    }
  }

  /**
   * Transforms a Mastra vector filter into an Upstash-compatible filter string.
   * @param {UpstashVectorFilter} [filter] - The filter to transform.
   * @returns {string | undefined} The transformed filter string, or undefined if no filter is provided.
   */
  transformFilter(filter?: UpstashVectorFilter) {
    const translator = new UpstashFilterTranslator();
    return translator.translate(filter);
  }

  /**
   * Creates a new index. For Upstash, this is a no-op as indexes (known as namespaces in Upstash) are created on-the-fly.
   * @param {CreateIndexParams} _params - The parameters for creating the index (ignored).
   * @returns {Promise<void>} A promise that resolves when the operation is complete.
   */
  async createIndex(_params: CreateIndexParams): Promise<void> {
    this.logger.debug('No need to call createIndex for Upstash');
  }

  /**
   * Queries the vector index.
   * @param {QueryVectorParams} params - The parameters for the query operation. indexName is the namespace in Upstash.
   * @returns {Promise<QueryResult[]>} A promise that resolves to the query results.
   */
  async query({
    indexName: namespace,
    queryVector,
    sparseVector,
    topK = 10,
    filter,
    includeVector = false,
    fusionAlgorithm,
    queryMode,
  }: UpstashQueryVectorParams): Promise<QueryResult[]> {
    try {
      const ns = this.client.namespace(namespace);
      const filterString = this.transformFilter(filter);

      // Build query parameters - using type assertion to handle Upstash's complex union types
      const queryParams = {
        topK,
        includeVectors: includeVector,
        includeMetadata: true,
        ...(queryVector && { vector: queryVector }),
        ...(sparseVector && sparseVector.indices.length > 0 && { sparseVector }),
        ...(filterString && { filter: filterString }),
        ...(queryVector && sparseVector && sparseVector.indices.length > 0 && fusionAlgorithm && { fusionAlgorithm }),
        ...(queryMode && { queryMode }),
      };

      const results = await ns.query(queryParams);

      // Map the results to our expected format
      return (results || []).map(result => ({
        id: `${result.id}`,
        score: result.score,
        metadata: result.metadata,
        ...(includeVector && { vector: result.vector || [] }),
      }));
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_QUERY_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace, topK },
        },
        error,
      );
    }
  }

  /**
   * Lists all namespaces in the Upstash vector index, which correspond to indexes.
   * @returns {Promise<string[]>} A promise that resolves to a list of index names.
   */
  async listIndexes(): Promise<string[]> {
    try {
      const indexes = await this.client.listNamespaces();
      return indexes.filter(Boolean);
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_LIST_INDEXES_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  /**
   * Retrieves statistics about a vector index.
   *
   * @param {string} indexName - The name of the namespace to describe
   * @returns A promise that resolves to the index statistics including dimension, count and metric
   */
  async describeIndex({ indexName: namespace }: DescribeIndexParams): Promise<IndexStats> {
    try {
      const info = await this.client.info();

      return {
        dimension: info.dimension,
        count: info.namespaces?.[namespace]?.vectorCount || 0,
        metric: info?.similarityFunction?.toLowerCase() as 'cosine' | 'euclidean' | 'dotproduct',
      };
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_DESCRIBE_INDEX_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace },
        },
        error,
      );
    }
  }

  /**
   * Deletes an index (namespace).
   * @param {DeleteIndexParams} params - The parameters for the delete operation.
   * @returns {Promise<void>} A promise that resolves when the deletion is complete.
   */
  async deleteIndex({ indexName: namespace }: DeleteIndexParams): Promise<void> {
    try {
      await this.client.deleteNamespace(namespace);
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_DELETE_INDEX_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace },
        },
        error,
      );
    }
  }

  /**
   * Updates a vector by its ID with the provided vector and/or metadata.
   * @param indexName - The name of the namespace containing the vector.
   * @param id - The ID of the vector to update.
   * @param update - An object containing the vector and/or metadata to update.
   * @param update.vector - An optional array of numbers representing the new vector.
   * @param update.metadata - An optional record containing the new metadata.
   * @param sparseVector - An optional sparse vector for Hybrid index compatibility.
   * @returns A promise that resolves when the update is complete.
   * @throws Will throw an error if no updates are provided or if the update operation fails.
   */
  async updateVector({ indexName: namespace, id, update, sparseVector }: UpstashUpdateVectorParams): Promise<void> {
    try {
      if (!update.vector && !update.metadata) {
        throw new Error('No update data provided');
      }

      // The upstash client throws an exception as: 'This index requires dense vectors' when
      // only metadata is present in the update object.
      if (!update.vector && update.metadata) {
        throw new Error('Both vector and metadata must be provided for an update');
      }
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_UPDATE_VECTOR_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace, id },
        },
        error,
      );
    }

    try {
      const updatePayload: any = { id: id };
      if (update.vector) {
        updatePayload.vector = update.vector;
      }
      if (update.metadata) {
        updatePayload.metadata = update.metadata;
      }

      const points = {
        id: updatePayload.id,
        vector: updatePayload.vector,
        metadata: updatePayload.metadata,
        ...(sparseVector && { sparseVector }),
      };

      await this.client.upsert(points, {
        namespace,
      });
    } catch (error) {
      throw new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_UPDATE_VECTOR_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace, id },
        },
        error,
      );
    }
  }

  /**
   * Deletes a vector by its ID.
   * @param indexName - The name of the namespace containing the vector.
   * @param id - The ID of the vector to delete.
   * @returns A promise that resolves when the deletion is complete.
   * @throws Will throw an error if the deletion operation fails.
   */
  async deleteVector({ indexName: namespace, id }: DeleteVectorParams): Promise<void> {
    try {
      await this.client.delete(id, {
        namespace,
      });
    } catch (error) {
      const mastraError = new MastraError(
        {
          id: 'STORAGE_UPSTASH_VECTOR_DELETE_VECTOR_FAILED',
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { namespace, id },
        },
        error,
      );
      this.logger?.error(mastraError.toString());
    }
  }
}
