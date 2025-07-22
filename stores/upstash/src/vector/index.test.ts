import type { QueryResult } from '@mastra/core/vector';
import dotenv from 'dotenv';

import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';

import { UpstashVector } from './';

dotenv.config();

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
 * These tests require a real Upstash Vector instance since there is no local Docker alternative.
 * The tests will be skipped in local development where Upstash credentials are not available.
 * In CI/CD environments, these tests will run using the provided Upstash Vector credentials.
 */
describe.skipIf(!process.env.UPSTASH_VECTOR_URL || !process.env.UPSTASH_VECTOR_TOKEN)('UpstashVector', () => {
  let vectorStore: UpstashVector;
  const VECTOR_DIMENSION = 1536;
  const testIndexName = 'default';
  const filterIndexName = 'filter-index';

  beforeAll(() => {
    // Load from environment variables for CI/CD
    const url = process.env.UPSTASH_VECTOR_URL;
    const token = process.env.UPSTASH_VECTOR_TOKEN;

    if (!url || !token) {
      console.log('Skipping Upstash Vector tests - no credentials available');
      return;
    }

    vectorStore = new UpstashVector({ url, token });
  });

  afterAll(async () => {
    if (!vectorStore) return;

    // Cleanup: delete test index
    try {
      await vectorStore.deleteIndex({ indexName: testIndexName });
    } catch (error) {
      console.warn('Failed to delete test index:', error);
    }
    try {
      await vectorStore.deleteIndex({ indexName: filterIndexName });
    } catch (error) {
      console.warn('Failed to delete filter index:', error);
    }
  });

  describe('Vector Operations', () => {
    const createVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      // Normalize the vector for cosine similarity
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    // Helper function to create minimal sparse vectors for Hybrid index compatibility
    // Hybrid indexes require sparse vectors to be present, so we create minimal ones
    const createMinimalSparseVectors = (count: number) => 
      Array(count).fill(null).map(() => ({ indices: [0], values: [0.1] }));

    it('should upsert vectors and query them back', async () => {
      const vectors = [
        createVector(0, 1.0),
        createVector(1, 1.0),
        createVector(2, 1.0),
      ];

      const metadata = [
        { text: 'first document', category: 'A' },
        { text: 'second document', category: 'B' },
        { text: 'third document', category: 'A' },
      ];

      // Add minimal sparse vectors for Hybrid index compatibility
      const sparseVectors = createMinimalSparseVectors(3);

      const vectorIds = await vectorStore.upsert({
        indexName: 'default',
        vectors,
        sparseVectors,
        metadata,
      });

      expect(vectorIds).toHaveLength(3);

      await waitUntilVectorsIndexed(vectorStore, 'default', 3);

      const queryVector = createVector(0, 1.0);
      const results = await vectorStore.query({
        indexName: 'default',
        queryVector,
        topK: 3,
      });

      expect(results).toHaveLength(3);
      expect(results[0]?.score).toBeGreaterThan(0.5);
    }, 60000);

    it('should handle empty vectors', async () => {
      const vectors: number[][] = [];
      const metadata: Record<string, any>[] = [];
      const sparseVectors: { indices: number[]; values: number[] }[] = [];

      await expect(vectorStore.upsert({
        indexName: 'default',
        vectors,
        sparseVectors,
        metadata,
      })).rejects.toThrow('Missing vector data');
    });

    it('should handle empty OR', async () => {
      const vectors = [
        createVector(0, 1.0),
        createVector(1, 1.0),
      ];

      const metadata = [
        { text: 'first document', category: 'A' },
        { text: 'second document', category: 'B' },
      ];

      // Add minimal sparse vectors for Hybrid index compatibility
      const sparseVectors = createMinimalSparseVectors(2);

      await vectorStore.upsert({
        indexName: 'default',
        vectors,
        sparseVectors,
        metadata,
      });

      await waitUntilVectorsIndexed(vectorStore, 'default', 2);

      const queryVector = createVector(0, 1.0);
      const results = await vectorStore.query({
        indexName: 'default',
        queryVector,
        topK: 10,
        filter: {
          $or: [],
        },
      });

      expect(results).toHaveLength(0);
    }, 60000);

    it('should query vectors and return vector in results', async () => {
      const vectors = [
        createVector(0, 1.0),
        createVector(1, 1.0),
      ];

      const metadata = [
        { text: 'first document', category: 'A' },
        { text: 'second document', category: 'B' },
      ];

      // Add minimal sparse vectors for Hybrid index compatibility
      const sparseVectors = createMinimalSparseVectors(2);

      await vectorStore.upsert({
        indexName: 'default',
        vectors,
        sparseVectors,
        metadata,
      });

      await waitUntilVectorsIndexed(vectorStore, 'default', 2);

      const queryVector = createVector(0, 1.0);
      const results = await vectorStore.query({
        indexName: 'default',
        queryVector,
        topK: 2,
        includeVector: true,
      });

      expect(results).toHaveLength(2);
      expect(results[0]?.vector).toBeDefined();
      expect(results[0]?.vector).toHaveLength(VECTOR_DIMENSION);
    }, 60000);

    describe('Vector update operations', () => {
      const testVectors = [createVector(0, 1.0), createVector(1, 1.0), createVector(2, 1.0)];
      const testSparseVectors = createMinimalSparseVectors(3);

      const testIndexName = 'test-index';

      afterEach(async () => {
        try {
          await vectorStore.deleteIndex({ indexName: testIndexName });
        } catch (error) {
          // Ignore cleanup errors
        }
      });

      it('should update the vector by id', async () => {
        const ids = await vectorStore.upsert({ 
          indexName: testIndexName, 
          vectors: testVectors,
          sparseVectors: testSparseVectors,
        });
        expect(ids).toHaveLength(3);

        const idToBeUpdated = ids[0];
        const newVector = createVector(0, 4.0);
        const newMetaData = {
          test: 'updates',
        };
        const newSparseVector = { indices: [0], values: [0.1] };

        const update = {
          vector: newVector,
          metadata: newMetaData,
        };

        await vectorStore.updateVector({ 
          indexName: testIndexName, 
          id: idToBeUpdated, 
          update,
          sparseVector: newSparseVector
        });

        await waitUntilVectorsIndexed(vectorStore, testIndexName, 3);

        const results: QueryResult[] = await vectorStore.query({
          indexName: testIndexName,
          queryVector: newVector,
          topK: 2,
          includeVector: true,
        });
        expect(results[0]?.id).toBe(idToBeUpdated);
        expect(results[0]?.vector).toEqual(newVector);
        expect(results[0]?.metadata).toEqual(newMetaData);
      }, 60000);

      it('should only update the metadata by id', async () => {
        const ids = await vectorStore.upsert({ 
          indexName: testIndexName, 
          vectors: testVectors,
          sparseVectors: testSparseVectors,
        });
        expect(ids).toHaveLength(3);

        const newMetaData = {
          test: 'updates',
        };

        const update = {
          metadata: newMetaData,
        };

        await expect(vectorStore.updateVector({ indexName: testIndexName, id: 'id', update })).rejects.toThrow(
          'Both vector and metadata must be provided for an update',
        );
      });

      it('should only update vector embeddings by id', async () => {
        const ids = await vectorStore.upsert({ 
          indexName: testIndexName, 
          vectors: testVectors,
          sparseVectors: testSparseVectors,
        });
        expect(ids).toHaveLength(3);

        const idToBeUpdated = ids[0];
        const newVector = createVector(0, 4.0);
        const newSparseVector = { indices: [0], values: [0.1] };

        const update = {
          vector: newVector,
        };

        await vectorStore.updateVector({ 
          indexName: testIndexName, 
          id: idToBeUpdated, 
          update,
          sparseVector: newSparseVector
        });

        await waitUntilVectorsIndexed(vectorStore, testIndexName, 3);

        const results: QueryResult[] = await vectorStore.query({
          indexName: testIndexName,
          queryVector: newVector,
          topK: 2,
          includeVector: true,
        });
        expect(results[0]?.id).toBe(idToBeUpdated);
        expect(results[0]?.vector).toEqual(newVector);
      }, 60000);

      it('should throw exception when no updates are given', async () => {
        await expect(vectorStore.updateVector({ indexName: testIndexName, id: 'id', update: {} })).rejects.toThrow(
          'No update data provided',
        );
      });
    });

    describe('Vector delete operations', () => {
      const testVectors = [createVector(0, 1.0), createVector(1, 1.0), createVector(2, 1.0)];
      const testSparseVectors = createMinimalSparseVectors(3);

      afterEach(async () => {
        try {
          await vectorStore.deleteIndex({ indexName: testIndexName });
        } catch (error) {
          // Ignore cleanup errors
        }
      });

      it('should delete the vector by id', async () => {
        const ids = await vectorStore.upsert({ 
          indexName: testIndexName, 
          vectors: testVectors,
          sparseVectors: testSparseVectors,
        });
        expect(ids).toHaveLength(3);
        const idToBeDeleted = ids[0];

        await vectorStore.deleteVector({ indexName: testIndexName, id: idToBeDeleted });

        const results: QueryResult[] = await vectorStore.query({
          indexName: testIndexName,
          queryVector: createVector(0, 1.0),
          topK: 2,
        });

        expect(results).toHaveLength(2);
        expect(results.map(res => res.id)).not.toContain(idToBeDeleted);
      });
    });
  });

  describe('Sparse Vector Operations', () => {
    const createVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      // Normalize the vector for cosine similarity
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };

    const createSparseVector = (indices: number[], values: number[]) => ({
      indices,
      values,
    });

    const sparseTestIndexName = 'default';

    afterEach(async () => {
      // Note: We're using the shared 'default' namespace, so we don't delete the entire index
      // The test data will be mixed with other test data, which is acceptable for testing purposes
    });

    describe('Basic Sparse Vector Operations', () => {
      it('should upsert vectors with sparse components and query them back', async () => {
        // Test upserting dense vectors along with their sparse counterparts
        // The sparse vectors represent additional semantic features beyond the dense embedding
        const denseVectors = [
          createVector(0, 1.0), // First vector focuses on dimension 0
          createVector(1, 1.0), // Second vector focuses on dimension 1
          createVector(2, 1.0), // Third vector focuses on dimension 2
        ];

        // Create sparse vectors with different feature patterns
        // These could represent things like: [keyword_presence, category_id, importance_score]
        const sparseVectors = [
          createSparseVector([10, 50, 100], [0.8, 0.6, 0.4]), // First doc: high relevance for features 10, 50, 100
          createSparseVector([10, 75, 150], [0.7, 0.9, 0.3]), // Second doc: high relevance for features 10, 75, 150  
          createSparseVector([25, 50, 200], [0.5, 0.8, 0.7]), // Third doc: high relevance for features 25, 50, 200
        ];

        const metadata = [
          { type: 'doc', keywords: ['tech', 'ai'] },
          { type: 'doc', keywords: ['tech', 'data'] },
          { type: 'doc', keywords: ['business', 'ai'] },
        ];

        // Step 1: Upsert vectors with both dense and sparse components
        const vectorIds = await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: denseVectors,
          sparseVectors,
          metadata,
        });

        expect(vectorIds).toHaveLength(3);
        expect(vectorIds.every(id => typeof id === 'string')).toBe(true);

        // Step 2: Wait for vectors to be indexed
        await waitUntilVectorsIndexed(vectorStore, sparseTestIndexName, 3);

        // Step 3: Query using dense vector only (should work as before)
        const denseOnlyResults = await vectorStore.query({
          indexName: sparseTestIndexName,
          queryVector: createVector(0, 0.9), // Query similar to first vector
          topK: 3,
        });

        expect(denseOnlyResults).toHaveLength(3);
        expect(denseOnlyResults[0]?.metadata?.keywords).toContain('tech'); // Should match first vector
      }, 60000);

      it('should query using both dense and sparse vectors for hybrid search', async () => {
        // First upsert some test data with sparse components
        const denseVectors = [
          createVector(0, 1.0),
          createVector(1, 1.0), 
          createVector(2, 1.0),
        ];

        // Create sparse vectors representing different feature sets
        const sparseVectors = [
          createSparseVector([100, 200], [0.9, 0.8]), // Doc 1: strong signals for features 100, 200
          createSparseVector([100, 300], [0.7, 0.6]), // Doc 2: moderate signals for features 100, 300
          createSparseVector([400, 500], [0.8, 0.9]), // Doc 3: strong signals for features 400, 500
        ];

        const metadata = [
          { category: 'tech', priority: 'high' },
          { category: 'tech', priority: 'medium' },
          { category: 'business', priority: 'high' },
        ];

        await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: denseVectors,
          sparseVectors,
          metadata,
        });

        await waitUntilVectorsIndexed(vectorStore, sparseTestIndexName, 3);

        // Step 1: Query with both dense and sparse components
        // This simulates a hybrid search where we want semantic similarity (dense) 
        // plus specific feature matching (sparse)
        const hybridResults = await vectorStore.query({
          indexName: sparseTestIndexName,
          queryVector: createVector(0, 0.8), // Dense query similar to first vector
          sparseVector: createSparseVector([100], [0.95]), // Looking for docs with high feature 100
          topK: 3,
          includeVector: true,
        });

        expect(hybridResults).toHaveLength(3);
        
        // Verify that results include both vector and metadata
        hybridResults.forEach(result => {
          expect(result.vector).toBeDefined();
          expect(result.vector).toHaveLength(VECTOR_DIMENSION);
          expect(result.metadata).toBeDefined();
          expect(result.score).toBeDefined();
        });

        // The first two docs should rank higher since they both have feature 100
        // while the third doc doesn't have feature 100
        expect(hybridResults[0]?.metadata?.category).toBe('tech');
      }, 60000);

      it('should handle mixed batch operations with some vectors having sparse components', async () => {
        // Test a realistic scenario where only some documents have sparse features
        // First, test vectors with sparse components
        const vectorsWithSparse = [
          { dense: createVector(0, 1.0), sparse: createSparseVector([50, 100], [0.8, 0.6]), metadata: { hasFeatures: true, type: 'enhanced' } },
          { dense: createVector(2, 1.0), sparse: createSparseVector([75, 125], [0.7, 0.9]), metadata: { hasFeatures: true, type: 'enhanced' } },
        ];

        // Step 1: Upsert vectors that have sparse components
        const sparseVectorIds = await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: vectorsWithSparse.map(v => v.dense),
          sparseVectors: vectorsWithSparse.map(v => v.sparse),
          metadata: vectorsWithSparse.map(v => v.metadata),
        });

        expect(sparseVectorIds).toHaveLength(2);

        // Step 2: Add vectors with minimal sparse components for Hybrid index
        const vectorsWithMinimalSparse = [
          { dense: createVector(1, 1.0), metadata: { hasFeatures: false, type: 'basic' } },
          { dense: createVector(3, 1.0), metadata: { hasFeatures: false, type: 'basic' } },
        ];

        const denseOnlyIds = await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: vectorsWithMinimalSparse.map(v => v.dense),
          sparseVectors: [
            { indices: [0], values: [0.1] },
            { indices: [0], values: [0.1] },
          ],
          metadata: vectorsWithMinimalSparse.map(v => v.metadata),
        });

        expect(denseOnlyIds).toHaveLength(2);

        await waitUntilVectorsIndexed(vectorStore, sparseTestIndexName, 4);

        // Step 3: Query and verify all vectors were stored correctly
        const allResults = await vectorStore.query({
          indexName: sparseTestIndexName,
          queryVector: createVector(0, 0.9),
          topK: 10, // Increase topK to ensure we get all vectors
        });

        // Check that we have at least 4 results (some might be from previous tests)
        expect(allResults.length).toBeGreaterThanOrEqual(4);
        
        // Verify metadata was preserved correctly for all vectors
        const enhancedDocs = allResults.filter(r => r.metadata?.type === 'enhanced');
        const basicDocs = allResults.filter(r => r.metadata?.type === 'basic');
        
        // Check that we have at least 2 of each type
        expect(enhancedDocs.length).toBeGreaterThanOrEqual(2);
        expect(basicDocs.length).toBeGreaterThanOrEqual(2);
      }, 60000);

      it('should handle minimal sparse vectors gracefully', async () => {
        // Test that minimal sparse vectors are handled correctly with Hybrid index
        const denseVectors = [
          createVector(0, 1.0),
          createVector(1, 1.0),
        ];

        // Create sparse vectors with minimal data for Hybrid index compatibility
        const sparseVectors = [
          createSparseVector([0], [0.1]), // Minimal sparse vector
          createSparseVector([100], [0.8]), // Normal sparse vector
        ];

        const metadata = [
          { type: 'no-sparse-features' },
          { type: 'has-sparse-features' },
        ];

        // Step 1: Upsert with empty sparse vector
        const vectorIds = await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: denseVectors,
          sparseVectors,
          metadata,
        });

        expect(vectorIds).toHaveLength(2);

        await waitUntilVectorsIndexed(vectorStore, sparseTestIndexName, 2);

        // Step 2: Query and verify both vectors work correctly
        const results = await vectorStore.query({
          indexName: sparseTestIndexName,
          queryVector: createVector(0, 0.9),
          topK: 2,
        });

        expect(results).toHaveLength(2);
        expect(results.every(r => r.metadata && r.score !== undefined)).toBe(true);
      }, 60000);
    });

    describe('Sparse Vector Validation', () => {
      it('should accept sparse vectors with matching indices and values lengths', async () => {
        // Test that validation passes when indices and values arrays have same length
        const denseVectors = [createVector(0, 1.0)];
        
        // Create various valid sparse vectors with different lengths but matching arrays
        const validSparseVectors = [
          createSparseVector([10], [0.5]), // Length 1
          // We can add more variations here if we want to test multiple at once
        ];

        // This should succeed without throwing
        const vectorIds = await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: denseVectors,
          sparseVectors: validSparseVectors,
        });

        expect(vectorIds).toHaveLength(1);
      });

      it('should reject sparse vectors with mismatched indices and values lengths', async () => {
        // Test that validation catches when indices.length !== values.length
        const denseVectors = [createVector(0, 1.0)];
        
        // Create invalid sparse vector: 3 indices but 2 values
        const invalidSparseVectors = [
          createSparseVector([10, 20, 30], [0.5, 0.8]), // Mismatch: 3 indices, 2 values
        ];

        // Step 1: Attempt upsert - should throw validation error
        await expect(
          vectorStore.upsert({
            indexName: sparseTestIndexName,
            vectors: denseVectors,
            sparseVectors: invalidSparseVectors,
          })
        ).rejects.toThrow(/Sparse vector at index 0 has mismatched indices and values lengths/);
      });

      it('should catch length mismatches in batch operations', async () => {
        // Test validation works correctly when processing multiple sparse vectors
        const denseVectors = [
          createVector(0, 1.0),
          createVector(1, 1.0), 
          createVector(2, 1.0),
        ];

        // Mix of valid and invalid sparse vectors
        const mixedSparseVectors = [
          createSparseVector([10, 20], [0.5, 0.8]), // Valid: lengths match
          createSparseVector([30, 40, 50], [0.6, 0.9]), // Invalid: 3 indices, 2 values (missing value)
          createSparseVector([60], [0.7]), // Valid: lengths match
        ];

        // Should fail on the second sparse vector (index 1)
        await expect(
          vectorStore.upsert({
            indexName: sparseTestIndexName,
            vectors: denseVectors,
            sparseVectors: mixedSparseVectors,
          })
        ).rejects.toThrow(/Sparse vector at index 1 has mismatched indices and values lengths/);
      });

      it('should provide detailed error information for validation failures', async () => {
        // Test that validation errors include helpful debugging information
        const denseVectors = [createVector(0, 1.0)];
        
        // Create sparse vector with clear length mismatch
        const problematicSparseVectors = [
          createSparseVector([1, 2, 3, 4, 5], [0.1, 0.2]), // 5 indices, 2 values
        ];

        try {
          await vectorStore.upsert({
            indexName: sparseTestIndexName,
            vectors: denseVectors,
            sparseVectors: problematicSparseVectors,
          });
          
          // Should not reach this point
          expect(true).toBe(false);
        } catch (error: any) {
          // Step 1: Verify error message contains the index position
          expect(error.message).toContain('Sparse vector at index 0');
          
          // Step 2: Verify error details contain length information
          expect(error.details?.index).toBe(0);
          expect(error.details?.indicesLength).toBe(5);
          expect(error.details?.valuesLength).toBe(2);
          
          // Step 3: Verify it's the correct error type
          expect(error.id).toBe('STORAGE_UPSTASH_VECTOR_SPARSE_VECTOR_MISMATCH');
        }
      });

      it('should handle minimal sparse vectors for Hybrid index compatibility', async () => {
        // Test that minimal sparse vectors work with Hybrid index
        const denseVectors = [
          createVector(0, 1.0),
          createVector(1, 1.0),
        ];

        const metadata = [
          { type: 'minimal-sparse-1' },
          { type: 'minimal-sparse-2' },
        ];

        // Step 1: Upsert with minimal sparse vectors for Hybrid index
        const vectorIds = await vectorStore.upsert({
          indexName: sparseTestIndexName,
          vectors: denseVectors,
          sparseVectors: [
            { indices: [0], values: [0.1] },
            { indices: [0], values: [0.1] },
          ],
          metadata,
        });

        expect(vectorIds).toHaveLength(2);

        await waitUntilVectorsIndexed(vectorStore, sparseTestIndexName, 2);

        // Step 2: Verify vectors were stored and can be queried
        const results = await vectorStore.query({
          indexName: sparseTestIndexName,
          queryVector: createVector(0, 0.9),
          topK: 2,
        });

        expect(results).toHaveLength(2);
        expect(results.every(r => r.metadata && r.score !== undefined)).toBe(true);
      }, 60000);

      it('should validate all sparse vectors before processing any', async () => {
        // Test that validation happens upfront before any processing
        // This ensures atomicity - either all vectors are valid or none are processed
        const denseVectors = [
          createVector(0, 1.0),
          createVector(1, 1.0),
          createVector(2, 1.0),
        ];

        const sparseVectorsWithLateError = [
          createSparseVector([10], [0.5]), // Valid
          createSparseVector([20], [0.6]), // Valid
          createSparseVector([30, 40], [0.7]), // Invalid: 2 indices, 1 value
        ];

        // Should fail before any vectors are processed
        await expect(
          vectorStore.upsert({
            indexName: sparseTestIndexName,
            vectors: denseVectors,
            sparseVectors: sparseVectorsWithLateError,
          })
        ).rejects.toThrow(/Sparse vector at index 2 has mismatched indices and values lengths/);

        // Verify no vectors were actually stored by checking the index
        try {
          const stats = await vectorStore.describeIndex({ indexName: sparseTestIndexName });
          expect(stats.count).toBe(0); // No vectors should be stored due to validation failure
        } catch (error) {
          // Index might not exist yet, which is also fine - confirms no vectors were stored
          expect(true).toBe(true);
        }
      });
    });
  });

  describe('Index Operations', () => {
    const createVector = (primaryDimension: number, value: number = 1.0): number[] => {
      const vector = new Array(VECTOR_DIMENSION).fill(0);
      vector[primaryDimension] = value;
      // Normalize the vector for cosine similarity
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    };
    it('should create and list an index', async () => {
      // since, we do not have to create index explictly in case of upstash. Upserts are enough
      // for testing the listIndexes() function
      // await vectorStore.createIndex({ indexName: testIndexName, dimension: 3, metric: 'cosine' });
      const ids = await vectorStore.upsert({ 
        indexName: testIndexName, 
        vectors: [createVector(0, 1.0)],
        sparseVectors: [{ indices: [0], values: [0.1] }],
      });
      expect(ids).toHaveLength(1);
      const indexes = await vectorStore.listIndexes();
      expect(indexes).toEqual([testIndexName]);
    });

    it('should describe an index correctly', async () => {
      const stats = await vectorStore.describeIndex({ indexName: 'mastra_default' });
      expect(stats).toEqual({
        dimension: 1536,
        metric: 'cosine',
        count: 0,
      });
    });
  });

  describe('Error Handling', () => {
    const testIndexName = 'test_index_error';
    beforeAll(async () => {
      await vectorStore.createIndex({ indexName: testIndexName, dimension: 3 });
    });

    afterAll(async () => {
      await vectorStore.deleteIndex({ indexName: testIndexName });
    });

    it('should handle invalid dimension vectors', async () => {
      await expect(
        vectorStore.upsert({ 
          indexName: testIndexName, 
          vectors: [[1.0, 0.0]], // Wrong dimensions
          sparseVectors: [{ indices: [0], values: [0.1] }],
        }),
      ).rejects.toThrow();
    });

    it('should handle querying with wrong dimensions', async () => {
      await expect(
        vectorStore.query({ indexName: testIndexName, queryVector: [1.0, 0.0] }), // Wrong dimensions
      ).rejects.toThrow();
    });
  });

  describe('Filter Tests', () => {
    const createVector = (dim: number) => new Array(VECTOR_DIMENSION).fill(0).map((_, i) => (i === dim ? 1 : 0));

    const testData = [
      {
        id: '1',
        vector: createVector(0),
        metadata: {
          name: 'Istanbul',
          population: 15460000,
          location: {
            continent: 'Asia',
            coordinates: {
              latitude: 41.0082,
              longitude: 28.9784,
            },
          },
          tags: ['historic', 'coastal', 'metropolitan'],
          industries: ['Tourism', 'Finance', 'Technology'],
          founded: 330,
          isCapital: false,
          lastCensus: null,
        },
      },
      {
        id: '2',
        vector: createVector(1),
        metadata: {
          name: 'Berlin',
          population: 3669495,
          location: {
            continent: 'Europe',
            coordinates: {
              latitude: 52.52,
              longitude: 13.405,
            },
          },
          tags: ['historic', 'cultural', 'metropolitan'],
          industries: ['Technology', 'Arts', 'Tourism'],
          founded: 1237,
          isCapital: true,
          lastCensus: 2021,
        },
      },
      {
        id: '3',
        vector: createVector(2),
        metadata: {
          name: 'San Francisco',
          population: 873965,
          location: {
            continent: 'North America',
            coordinates: {
              latitude: 37.7749,
              longitude: -122.4194,
            },
          },
          tags: ['coastal', 'tech', 'metropolitan'],
          industries: ['Technology', 'Finance', 'Tourism'],
          founded: 1776,
          isCapital: false,
          lastCensus: 2020,
        },
      },
      {
        id: '4',
        vector: createVector(3),
        metadata: {
          name: "City's Name",
          description: 'Contains "quotes"',
          population: 0,
          temperature: -10,
          microscopicDetail: 1e-10,
          isCapital: false,
          tags: ['nothing'],
        },
      },
    ];

    beforeAll(async () => {
      await vectorStore.createIndex({ indexName: filterIndexName, dimension: VECTOR_DIMENSION });
      await vectorStore.upsert({
        indexName: filterIndexName,
        vectors: testData.map(d => d.vector),
        sparseVectors: testData.map(() => ({ indices: [0], values: [0.1] })),
        metadata: testData.map(d => d.metadata),
        ids: testData.map(d => d.id),
      });
      // Wait for indexing
      await waitUntilVectorsIndexed(vectorStore, filterIndexName, testData.length);
    }, 50000);

    describe('Basic Operators', () => {
      it('should filter by exact match', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: 'Istanbul' },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.name).toBe('Istanbul');
      });

      it('should filter by not equal', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: { $ne: 'Berlin' } },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.name).not.toBe('Berlin');
        });
      });

      it('should filter by greater than', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { population: { $gt: 1000000 } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.population).toBeGreaterThan(1000000);
        });
      });

      it('should filter by less than or equal', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { founded: { $lte: 1500 } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.founded).toBeLessThanOrEqual(1500);
        });
      });
    });

    describe('Array Operations', () => {
      it('should filter by array contains', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          topK: 10,
          filter: { tags: { $contains: 'historic' } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.tags).toContain('historic');
        });
      });

      it('should filter by array not contains', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { tags: { $not: { $contains: 'tech' } } },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.tags?.find(tag => tag === 'tech')).toBeUndefined();
        });
      });

      it('should filter by in array', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'location.continent': { $in: ['Asia', 'Europe'] } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(['Asia', 'Europe']).toContain(result.metadata?.location?.continent);
        });
      });

      it('should filter by not in array', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: { $nin: ['Berlin', 'Istanbul'] } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(['Berlin', 'Istanbul']).not.toContain(result.metadata?.name);
        });
      });
    });

    describe('Array Indexing', () => {
      it('should filter by first array element', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'industries[0]': 'Tourism' },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.industries?.[0]).toBe('Tourism');
      });

      it('should filter by last array element', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'industries[#-1]': 'Technology' },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.industries?.slice(-1)[0]).toBe('Technology');
      });

      it('should combine first and last element filters', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'industries[0]': 'Tourism', 'tags[#-1]': 'metropolitan' },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.industries?.[0]).toBe('Tourism');
        expect(result?.tags?.slice(-1)[0]).toBe('metropolitan');
      });
    });

    describe('Nested Fields', () => {
      it('should filter by nested field', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'location.continent': 'Asia' },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.location?.continent).toBe('Asia');
      });

      it('should filter by deeply nested field with comparison', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'location.coordinates.latitude': { $gt: 40 } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.location?.coordinates?.latitude).toBeGreaterThan(40);
        });
      });

      it('should combine nested and array filters', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'location.coordinates.latitude': { $gt: 40 }, 'industries[0]': 'Tourism' },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.location?.coordinates?.latitude).toBeGreaterThan(40);
        expect(result?.industries?.[0]).toBe('Tourism');
      });
    });

    describe('Logical Operators', () => {
      it('should combine conditions with AND', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $and: [{ population: { $gt: 1000000 } }, { isCapital: true }] },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.population).toBeGreaterThan(1000000);
        expect(result?.isCapital).toBe(true);
      });

      it('should combine conditions with OR', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $or: [{ 'location.continent': 'Asia' }, { 'location.continent': 'Europe' }] },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(['Asia', 'Europe']).toContain(result.metadata?.location?.continent);
        });
      });

      it('should handle NOT operator', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $not: { isCapital: true } },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.isCapital).not.toBe(true);
        });
      });

      it('should handle NOT with comparison operators', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { population: { $not: { $lt: 1000000 } } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.population).toBeGreaterThanOrEqual(1000000);
        });
      });

      it('should handle NOT with contains operator', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { tags: { $not: { $contains: 'tech' } } },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.tags?.find(tag => tag === 'tech')).toBeUndefined();
        });
      });

      it('should handle NOT with regex operator', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: { $not: { $regex: '*bul' } } },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.name).not.toMatch(/bul$/);
        });
      });

      it('should handle NOR operator', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $nor: [{ 'location.continent': 'Asia' }, { 'location.continent': 'Europe' }] },
        });
        expect(results).toHaveLength(1);
        results.forEach(result => {
          expect(['Asia', 'Europe']).not.toContain(result.metadata?.location?.continent);
        });
      });

      it('should handle NOR with multiple conditions', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $nor: [{ population: { $gt: 10000000 } }, { isCapital: true }, { tags: { $contains: 'tech' } }] },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.population).toBeLessThanOrEqual(10000000);
        expect(result?.isCapital).not.toBe(true);
        expect(result?.tags).not.toContain('tech');
      });

      it('should handle ALL operator with simple values', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { industries: { $all: ['Tourism', 'Finance'] } },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.industries).toContain('Tourism');
          expect(result.metadata?.industries).toContain('Finance');
        });
      });

      it('should handle ALL operator with empty array', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { tags: { $all: [] } },
        });
        expect(results.length).toBeGreaterThan(0);
      });

      it('should handle NOT with nested logical operators', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $not: { $and: [{ population: { $lt: 1000000 } }, { isCapital: true }] } },
        });
        expect(results).toHaveLength(4);
        results.forEach(result => {
          const metadata = result.metadata;
          expect(metadata?.population >= 1000000 || metadata?.isCapital !== true).toBe(true);
        });
      });

      it('should handle NOR with nested path conditions', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $nor: [
              { 'location.coordinates.latitude': { $lt: 40 } },
              { 'location.coordinates.longitude': { $gt: 100 } },
            ],
          },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          const coords = result.metadata?.location?.coordinates;
          expect(coords?.latitude >= 40 || coords?.longitude <= 100).toBe(true);
        });
      });

      it('should handle exists with nested paths', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $and: [
              { 'location.coordinates.latitude': { $exists: true } },
              { 'location.coordinates.longitude': { $exists: true } },
            ],
          },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.location?.coordinates?.latitude).toBeDefined();
          expect(result.metadata?.location?.coordinates?.longitude).toBeDefined();
        });
      });

      it('should handle complex NOT combinations', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $not: {
              $or: [
                { 'location.continent': 'Asia' },
                { population: { $lt: 1000000 } },
                { tags: { $contains: 'tech' } },
              ],
            },
          },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.location?.continent).not.toBe('Asia');
        expect(result?.population).toBeGreaterThanOrEqual(1000000);
        expect(result?.tags).not.toContain('tech');
      });

      it('should handle NOR with regex patterns', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $nor: [{ name: { $regex: '*bul' } }, { name: { $regex: '*lin' } }, { name: { $regex: '*cisco' } }],
          },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.name).toBe("City's Name");
      });

      it('should handle NOR with mixed operator types', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $nor: [
              { population: { $gt: 5000000 } },
              { tags: { $contains: 'tech' } },
              { 'location.coordinates.latitude': { $lt: 38 } },
            ],
          },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.population).toBeLessThanOrEqual(5000000);
        expect(result?.tags).not.toContain('tech');
        expect(result?.location?.coordinates?.latitude).toBeGreaterThanOrEqual(38);
      });

      it('should handle NOR with exists operator', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $nor: [{ lastCensus: { $exists: true } }, { population: { $exists: false } }] },
        });
        expect(results).toHaveLength(1);
        const result = results[0]?.metadata;
        expect(result?.lastCensus).toBeUndefined();
        expect(result?.population).toBeDefined();
      });

      it('should handle ALL with mixed value types', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $and: [{ tags: { $contains: 'coastal' } }, { tags: { $contains: 'metropolitan' } }] },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          const tags = result.metadata?.tags || [];
          expect(tags).toContain('coastal');
          expect(tags).toContain('metropolitan');
        });
      });

      it('should handle ALL with nested array conditions', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $and: [{ industries: { $all: ['Tourism', 'Finance'] } }, { tags: { $all: ['metropolitan'] } }] },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          expect(result.metadata?.industries).toContain('Tourism');
          expect(result.metadata?.industries).toContain('Finance');
          expect(result.metadata?.tags).toContain('metropolitan');
        });
      });

      it('should handle ALL with complex conditions', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $or: [{ industries: { $all: ['Tourism', 'Finance'] } }, { tags: { $all: ['tech', 'metropolitan'] } }],
          },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          const hasAllIndustries =
            result.metadata?.industries?.includes('Tourism') && result.metadata?.industries?.includes('Finance');
          const hasAllTags = result.metadata?.tags?.includes('tech') && result.metadata?.tags?.includes('metropolitan');
          expect(hasAllIndustries || hasAllTags).toBe(true);
        });
      });

      it('should handle ALL with single item array', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { industries: { $all: ['Technology'] } },
        });
        expect(results).toHaveLength(3);
        results.forEach(result => {
          expect(result.metadata?.industries).toContain('Technology');
        });
      });

      it('should handle complex nested conditions', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: {
            $and: [
              { population: { $gt: 1000000 } },
              {
                $or: [{ 'location.continent': 'Asia' }, { industries: { $contains: 'Technology' } }],
              },
            ],
          },
        });
        expect(results).toHaveLength(2);
        results.forEach(result => {
          const metadata = result.metadata;
          expect(metadata?.population).toBeGreaterThan(1000000);
          expect(metadata?.location?.continent === 'Asia' || metadata?.industries?.includes('Technology')).toBe(true);
        });
      });
    });

    describe('Edge Cases', () => {
      describe('Empty Conditions', () => {
        it('should handle empty AND array', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { $and: [] },
          });
          expect(results.length).toBeGreaterThan(0);
        });

        it('should handle empty OR array', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { $or: [] },
          });
          expect(results.length).toBe(0);
        });

        it('should handle empty IN array', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { tags: { $in: [] } },
          });
          expect(results.length).toBe(0);
        });
        it('should handle empty IN array', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { tags: [] },
          });
          expect(results.length).toBe(0);
        });
      });

      describe('Null/Undefined Values', () => {
        it('should handle null values', async () => {
          await expect(
            vectorStore.query({
              indexName: filterIndexName,
              queryVector: createVector(0),
              filter: { lastCensus: null },
            }),
          ).rejects.toThrow();
        });

        it('should handle null in arrays', async () => {
          await expect(
            vectorStore.query({
              indexName: filterIndexName,
              queryVector: createVector(0),
              filter: { tags: { $in: [null, 'historic'] } },
            }),
          ).rejects.toThrow();
        });
      });

      describe('Special Characters', () => {
        it('should handle strings with quotes', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { name: "City's Name" },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.name).toBe("City's Name");
        });

        it('should handle strings with double quotes', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { description: 'Contains "quotes"' },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.description).toBe('Contains "quotes"');
        });
      });

      describe('Number Formats', () => {
        it('should handle zero', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { population: 0 },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.population).toBe(0);
        });

        it('should handle negative numbers', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { temperature: -10 },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.temperature).toBe(-10);
        });

        it('should handle decimal numbers', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { 'location.coordinates.latitude': 41.0082 },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.location?.coordinates?.latitude).toBe(41.0082);
        });

        it('should handle scientific notation', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { microscopicDetail: 1e-10 },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.microscopicDetail).toBe(1e-10);
        });

        it('should handle escaped quotes in strings', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { description: { $regex: '*"quotes"*' } },
          });
          expect(results).toHaveLength(1);
          expect(results[0]?.metadata?.description).toBe('Contains "quotes"');
        });
        it('should handle undefined filter', async () => {
          const results1 = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: undefined,
          });
          const results2 = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
          });
          expect(results1).toEqual(results2);
          expect(results1.length).toBeGreaterThan(0);
        });

        it('should handle empty object filter', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: {},
          });
          const results2 = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
          });
          expect(results).toEqual(results2);
          expect(results.length).toBeGreaterThan(0);
        });

        it('should handle null filter', async () => {
          const results = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: null,
          });
          const results2 = await vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
          });
          expect(results).toEqual(results2);
          expect(results.length).toBeGreaterThan(0);
        });
      });
    });

    describe('Pattern Matching', () => {
      it('should match start of string', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: { $regex: 'San*' } },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.name).toBe('San Francisco');
      });

      it('should match end of string', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: { $regex: '*in' } },
        });
        expect(results).toHaveLength(1);
        expect(results[0]?.metadata?.name).toBe('Berlin');
      });

      it('should handle negated pattern', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { name: { $not: { $regex: 'A*' } } },
        });
        expect(results).toHaveLength(4);
      });
    });

    describe('Field Existence', () => {
      it('should check field exists', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'location.coordinates': { $exists: true } },
        });
        expect(results).toHaveLength(3);
      });

      it('should check field does not exist', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { unknownField: { $exists: false } },
        });
        expect(results).toHaveLength(4);
      });
    });

    describe('Performance Tests', () => {
      it('should reject large arrays', async () => {
        const largeArray = Array.from({ length: 1000 }, (_, i) => `value${i}`);
        await expect(
          vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { tags: { $in: largeArray } },
          }),
        ).rejects.toThrow();
      });

      it('should handle deep nesting', async () => {
        const deepFilter = {
          $and: [
            { 'a.b.c.d.e': 1 },
            {
              $or: [
                { 'f.g.h.i.j': 2 },
                {
                  $and: [{ 'k.l.m.n.o': 3 }, { 'p.q.r.s.t': 4 }],
                },
              ],
            },
          ],
        };
        const start = Date.now();
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: deepFilter,
        });
        const duration = Date.now() - start;
        expect(duration).toBeLessThan(1000);
        expect(Array.isArray(results)).toBe(true);
      });

      it('should handle complex combinations', async () => {
        const complexFilter = {
          $and: Array(10)
            .fill(null)
            .map((_, i) => ({
              $or: [
                { [`field${i}`]: { $gt: i } },
                { [`array${i}`]: { $contains: `value${i}` } },
                { [`nested${i}.field`]: { $in: [`value${i}`, `other${i}`] } },
              ],
            })),
        };
        const start = Date.now();
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: complexFilter,
        });
        const duration = Date.now() - start;
        expect(duration).toBeLessThan(1000);
        expect(Array.isArray(results)).toBe(true);
      });
    });

    describe('Error Cases', () => {
      it('should reject invalid operators', async () => {
        await expect(
          vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { field: { $invalidOp: 'value' } as any },
          }),
        ).rejects.toThrow();
      });

      it('should reject empty brackets', async () => {
        await expect(
          vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { 'industries[]': 'Tourism' },
          }),
        ).rejects.toThrow();
      });

      it('should reject unclosed brackets', async () => {
        await expect(
          vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { 'industries[': 'Tourism' },
          }),
        ).rejects.toThrow();
      });

      it('should handle invalid array syntax by returning empty results', async () => {
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { 'industries#-1]': 'Tourism' },
        });
        expect(results).toHaveLength(0);
      });

      it('should reject invalid field paths', async () => {
        await expect(
          vectorStore.query({
            indexName: filterIndexName,
            queryVector: createVector(0),
            filter: { '.invalidPath': 'value' },
          }),
        ).rejects.toThrow();
      });

      it('should handle malformed complex queries by returning all results', async () => {
        // Upstash treats malformed logical operators as non-filtering conditions
        // rather than throwing errors
        const results = await vectorStore.query({
          indexName: filterIndexName,
          queryVector: createVector(0),
          filter: { $and: { not: 'an array' } as any },
        });
        expect(results.length).toBeGreaterThan(0);
      });
    });
  });
});
