import { UpstashVector } from './index';
import type { QueryResult } from '@mastra/core/vector';
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { FusionAlgorithm } from '@upstash/vector';
import dotenv from 'dotenv';

dotenv.config();

/**
 * Helper function to wait for vectors to be indexed
 */
function waitUntilVectorsIndexed(vector: UpstashVector, indexName: string, expectedCount: number) {
  return new Promise((resolve, reject) => {
    const maxAttempts = 30;
    let attempts = 0;
    const interval = setInterval(async () => {
      try {
        const stats = await vector.describeIndex({ indexName });
        if (stats && stats.count >= expectedCount) {
          clearInterval(interval);
          resolve(true);
        }
        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Timeout waiting for vectors to be indexed'));
        }
      } catch (error) {
        console.log(error);
      }
    }, 5000);
  });
}

/**
 * These tests require a real Upstash Vector Hybrid index instance.
 * The tests will be skipped in local development where Upstash credentials are not available.
 * In CI/CD environments, these tests will run using the provided Upstash Vector credentials.
 */
describe.skipIf(!process.env.UPSTASH_VECTOR_URL || !process.env.UPSTASH_VECTOR_TOKEN)('UpstashVector Hybrid Index', () => {
  let vectorStore: UpstashVector;
  const VECTOR_DIMENSION = 1536;
  const hybridIndexName = 'default';

  beforeAll(() => {
    // Load from environment variables for CI/CD
    const url = process.env.UPSTASH_VECTOR_URL;
    const token = process.env.UPSTASH_VECTOR_TOKEN;

    if (!url || !token) {
      console.log('Skipping Upstash Vector Hybrid tests - no credentials available');
      return;
    }

    vectorStore = new UpstashVector({ url, token });
  });

  afterAll(async () => {
    if (!vectorStore) return;

    // Cleanup: delete hybrid test index
    try {
      await vectorStore.deleteIndex({ indexName: hybridIndexName });
    } catch (error) {
      console.warn('Failed to delete hybrid test index:', error);
    }
  });

  beforeEach(async () => {
    // Clean up any existing data before each test
    try {
      await vectorStore.deleteIndex({ indexName: hybridIndexName });
    } catch (error) {
      // Ignore errors if index doesn't exist
    }
  });

  describe('Basic Hybrid Operations', () => {
    // Helper function to create a normalized dense vector
    const createDenseVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      // Normalize the vector for cosine similarity
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    // Helper function to create sparse vectors with meaningful features
    const createSparseVector = (indices: number[], values: number[]) => ({
      indices,
      values,
    });

    // Helper function to create minimal sparse vectors for Hybrid index compatibility
    const createMinimalSparseVector = () => ({ indices: [0], values: [0.1] });

    it('should upsert vectors with both dense and sparse components', async () => {
      const denseVectors = [
        createDenseVector(0, 1.0),
        createDenseVector(1, 1.0),
        createDenseVector(2, 1.0),
      ];

      const sparseVectors = [
        createSparseVector([100, 200], [0.8, 0.6]), // Document has features 100, 200
        createSparseVector([150, 250], [0.7, 0.9]), // Document has features 150, 250
        createSparseVector([300], [0.5]), // Document has feature 300
      ];

      const metadata = [
        { type: 'tech-doc', category: 'programming' },
        { type: 'tech-doc', category: 'ai' },
        { type: 'tech-doc', category: 'data' },
      ];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(3);
      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 3);

      // Query with dense vector only
      const denseResults = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9),
        topK: 3,
      });

      expect(denseResults).toHaveLength(3);
      expect(denseResults[0]?.metadata?.category).toBe('programming');
    }, 60000);

    it('should query using both dense and sparse vectors for hybrid search', async () => {
      const denseVectors = [
        createDenseVector(0, 1.0),
        createDenseVector(1, 1.0),
      ];

      const sparseVectors = [
        createSparseVector([100], [0.8]), // Document has feature 100
        createSparseVector([200], [0.6]), // Document has feature 200
      ];

      const metadata = [
        { type: 'tech-doc', features: ['feature-100'] },
        { type: 'tech-doc', features: ['feature-200'] },
      ];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(2);
      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 2);

      // Query with both dense and sparse vectors
      const hybridResults = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9),
        sparseVector: createSparseVector([100], [0.8]),
        topK: 2,
      });

      expect(hybridResults).toHaveLength(2);
      // The first doc should rank higher since it has feature 100
      expect(hybridResults[0]?.metadata?.features).toContain('feature-100');
    }, 60000);

    it('should handle minimal sparse vectors for Hybrid index compatibility', async () => {
      const denseVectors = [
        createDenseVector(0, 1.0),
        createDenseVector(1, 1.0),
      ];

      const metadata = [
        { type: 'minimal-sparse-1' },
        { type: 'minimal-sparse-2' },
      ];

      // Use minimal sparse vectors for Hybrid index compatibility
      const minimalSparseVectors = [
        createMinimalSparseVector(),
        createMinimalSparseVector(),
      ];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors: minimalSparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(2);
      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 2);

      // Verify vectors were stored and can be queried
      const results = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9),
        topK: 2,
      });

      expect(results).toHaveLength(2);
      expect(results.every(r => r.metadata && r.score !== undefined)).toBe(true);
    }, 60000);
  });

  describe('Sparse Vector Requirements', () => {
    const createDenseVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    const createMinimalSparseVector = () => ({ indices: [0], values: [0.1] });

    it('should reject upsert operations without sparse vectors', async () => {
      const denseVectors = [createDenseVector(0, 1.0)];
      const metadata = [{ type: 'no-sparse' }];

      await expect(
        vectorStore.upsert({
          indexName: hybridIndexName,
          vectors: denseVectors,
          metadata,
        })
      ).rejects.toThrow('This index requires sparse vectors');
    });

    it('should validate sparse vector format (indices/values matching)', async () => {
      const denseVectors = [createDenseVector(0, 1.0)];
      const invalidSparseVectors = [{ indices: [1, 2], values: [0.5] }]; // Mismatched lengths

      await expect(
        vectorStore.upsert({
          indexName: hybridIndexName,
          vectors: denseVectors,
          sparseVectors: invalidSparseVectors,
        })
      ).rejects.toThrow(/Sparse vector at index 0 has mismatched indices and values lengths/);
    });

    it('should reject empty sparse vectors', async () => {
      const denseVectors = [createDenseVector(0, 1.0)];
      const emptySparseVectors = [{ indices: [], values: [] }];

      // Empty sparse vectors should be rejected for Hybrid indexes
      await expect(
        vectorStore.upsert({
          indexName: hybridIndexName,
          vectors: denseVectors,
          sparseVectors: emptySparseVectors,
        })
      ).rejects.toThrow();
    });
  });

  describe('Update Operations', () => {
    const createDenseVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    const createMinimalSparseVector = () => ({ indices: [0], values: [0.1] });

    it('should update dense vector only (with minimal sparse)', async () => {
      // First, upsert a vector
      const denseVectors = [createDenseVector(0, 1.0)];
      const sparseVectors = [createMinimalSparseVector()];
      const metadata = [{ type: 'original' }];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(1);

      // Update only the dense vector (must include minimal sparse vector)
      const newDenseVector = createDenseVector(1, 2.0);
      const newSparseVector = createMinimalSparseVector();

      await vectorStore.updateVector({
        indexName: hybridIndexName,
        id: vectorIds[0]!,
        update: { vector: newDenseVector },
        sparseVector: newSparseVector,
      });

      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 1);

      // Query to verify the update
      const results = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: newDenseVector,
        topK: 1,
        includeVector: true,
      });

      expect(results).toHaveLength(1);
      expect(results[0]?.vector).toEqual(newDenseVector);
    }, 60000);

    it('should update metadata only (with minimal sparse)', async () => {
      // First, upsert a vector
      const denseVectors = [createDenseVector(0, 1.0)];
      const sparseVectors = [createMinimalSparseVector()];
      const metadata = [{ type: 'original' }];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(1);

      // Update only metadata (must include both vector and sparse vector for Hybrid index)
      const newMetadata = { type: 'updated' };
      const newSparseVector = createMinimalSparseVector();

      await vectorStore.updateVector({
        indexName: hybridIndexName,
        id: vectorIds[0]!,
        update: { 
          vector: denseVectors[0]!, // Must include original vector
          metadata: newMetadata 
        },
        sparseVector: newSparseVector,
      });

      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 1);

      // Query to verify the update
      const results = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: denseVectors[0]!,
        topK: 1,
      });

      expect(results).toHaveLength(1);
      expect(results[0]?.metadata?.type).toBe('updated');
    }, 60000);

    it('should reject updates without sparse vectors', async () => {
      // First, upsert a vector
      const denseVectors = [createDenseVector(0, 1.0)];
      const sparseVectors = [createMinimalSparseVector()];
      const metadata = [{ type: 'original' }];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(1);

      // Try to update without sparse vector
      await expect(
        vectorStore.updateVector({
          indexName: hybridIndexName,
          id: vectorIds[0]!,
          update: { vector: createDenseVector(1, 2.0) },
        })
      ).rejects.toThrow('This index requires sparse vectors');
    });
  });

  describe('Query Operations', () => {
    const createDenseVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    const createSparseVector = (indices: number[], values: number[]) => ({
      indices,
      values,
    });

    beforeEach(async () => {
      // Setup test data
      const denseVectors = [
        createDenseVector(0, 1.0),
        createDenseVector(1, 1.0),
        createDenseVector(2, 1.0),
      ];

      const sparseVectors = [
        createSparseVector([100, 200], [0.8, 0.6]),
        createSparseVector([150, 250], [0.7, 0.9]),
        createSparseVector([300], [0.5]),
      ];

      const metadata = [
        { type: 'query-test', features: ['100', '200'] },
        { type: 'query-test', features: ['150', '250'] },
        { type: 'query-test', features: ['300'] },
      ];

      await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 3);
    });

    it('should perform dense-only queries (with sparse vectors in index)', async () => {
      const denseVectors = [
        createDenseVector(0, 1.0),
        createDenseVector(1, 1.0),
        createDenseVector(2, 1.0),
      ];

      const sparseVectors = [
        createSparseVector([100], [0.8]),
        createSparseVector([200], [0.6]),
        createSparseVector([300], [0.4]),
      ];

      const metadata = [
        { type: 'doc-1', features: ['100'] },
        { type: 'doc-2', features: ['200'] },
        { type: 'doc-3', features: ['300'] },
      ];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(3);
      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 3);

      // Query with dense vector only (Hybrid index still requires sparse vector in query)
      const results = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9),
        sparseVector: createSparseVector([100], [0.8]), // Still need sparse vector for Hybrid
        topK: 3,
      });

      expect(results).toHaveLength(3);
      // Check that we get results, but don't assume specific ordering
      expect(results[0]?.metadata).toBeDefined();
    }, 60000);

    it('should perform sparse-only queries (with dense vectors in index)', async () => {
      // Use the existing data from previous tests or create new data
      const denseVectors = [
        createDenseVector(0, 1.0),
        createDenseVector(1, 1.0),
        createDenseVector(2, 1.0),
      ];

      const sparseVectors = [
        createSparseVector([100], [0.8]),
        createSparseVector([200], [0.6]),
        createSparseVector([300], [0.4]),
      ];

      const metadata = [
        { type: 'doc-1', features: ['100'] },
        { type: 'doc-2', features: ['200'] },
        { type: 'doc-3', features: ['300'] },
      ];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(3);
      await waitUntilVectorsIndexed(vectorStore, hybridIndexName, 3);

      const results = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9), // Still need dense vector for Hybrid
        sparseVector: createSparseVector([100], [0.8]),
        topK: 3,
      });

      expect(results).toHaveLength(3);
      // Check that we get results, but don't assume specific ordering
      expect(results[0]?.metadata).toBeDefined();
    }, 60000);

    it('should handle fusion algorithms', async () => {
      // Test RRF (Reciprocal Rank Fusion)
      const rrfResults = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9),
        sparseVector: createSparseVector([100], [0.8]),
        topK: 3,
        fusionAlgorithm: FusionAlgorithm.RRF,
      });

      expect(rrfResults).toHaveLength(3);

      // Test DBSF (Dense Before Sparse Fusion)
      const dbsfResults = await vectorStore.query({
        indexName: hybridIndexName,
        queryVector: createDenseVector(0, 0.9),
        sparseVector: createSparseVector([100], [0.8]),
        topK: 3,
        fusionAlgorithm: FusionAlgorithm.DBSF,
      });

      expect(dbsfResults).toHaveLength(3);
    }, 60000);
  });

  describe('Error Handling', () => {
    const createDenseVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    it('should handle missing sparse vectors in upsert', async () => {
      const denseVectors = [createDenseVector(0, 1.0)];
      const metadata = [{ type: 'error-test' }];

      await expect(
        vectorStore.upsert({
          indexName: hybridIndexName,
          vectors: denseVectors,
          metadata,
        })
      ).rejects.toThrow('This index requires sparse vectors');
    });

    it('should handle missing sparse vectors in update', async () => {
      // First create a vector
      const denseVectors = [createDenseVector(0, 1.0)];
      const sparseVectors = [{ indices: [0], values: [0.1] }];
      const metadata = [{ type: 'error-test' }];

      const vectorIds = await vectorStore.upsert({
        indexName: hybridIndexName,
        vectors: denseVectors,
        sparseVectors,
        metadata,
      });

      // Try to update without sparse vector
      await expect(
        vectorStore.updateVector({
          indexName: hybridIndexName,
          id: vectorIds[0]!,
          update: { vector: createDenseVector(1, 2.0) },
        })
      ).rejects.toThrow('This index requires sparse vectors');
    });

    it('should handle invalid sparse vector format', async () => {
      const denseVectors = [createDenseVector(0, 1.0)];
      const invalidSparseVectors = [{ indices: [1, 2], values: [0.5] }];

      await expect(
        vectorStore.upsert({
          indexName: hybridIndexName,
          vectors: denseVectors,
          sparseVectors: invalidSparseVectors,
        })
      ).rejects.toThrow(/Sparse vector at index 0 has mismatched indices and values lengths/);
    });
  });
}); 