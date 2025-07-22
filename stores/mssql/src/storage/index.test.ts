import { createTestSuite } from '@internal/storage-test-utils';
import { describe, expect, it, vi } from 'vitest';

import { MSSQLStore } from '.';
import type { MSSQLConfig } from '.';

const TEST_CONFIG: MSSQLConfig = {
  server: process.env.MSSQL_HOST || 'localhost',
  port: Number(process.env.MSSQL_PORT) || 1433,
  database: process.env.MSSQL_DB || 'master',
  user: process.env.MSSQL_USER || 'sa',
  password: process.env.MSSQL_PASSWORD || 'Your_password123',
};

// const connectionString = `mssql://${TEST_CONFIG.user}:${TEST_CONFIG.password}@${TEST_CONFIG.server}:${TEST_CONFIG.port}/${TEST_CONFIG.database}`;

vi.setConfig({ testTimeout: 60_000, hookTimeout: 60_000 });

console.log('Not running MSSQL tests in CI. You can enable them if you want to test them locally.');
if (process.env.ENABLE_TESTS === 'true') {
  createTestSuite(new MSSQLStore(TEST_CONFIG));
} else {
  describe('MSSQLStore', () => {
    it('should be defined', () => {
      expect(MSSQLStore).toBeDefined();
    });
  });
}

// describe('MSSQLStore', () => {
//   let store: MSSQLStore;

//   beforeAll(async () => {
//     store = new MSSQLStore(TEST_CONFIG);
//     await store.init();
//   });

//   describe('Public Fields Access (MSSQL)', () => {
//     let testDB: MSSQLStore;

//     beforeAll(async () => {
//       testDB = new MSSQLStore(TEST_CONFIG);
//       await testDB.init();
//     });

//     afterAll(async () => {
//       try {
//         await testDB.close();
//       } catch { }
//     });

//     it('should expose pool field as public', () => {
//       expect(testDB.pool).toBeDefined();
//       // For mssql, db is likely a pool or connection
//       expect(typeof testDB.pool).toBe('object');
//       expect(typeof testDB.pool.request).toBe('function');
//     });

//     it('should allow direct database queries via public pool field', async () => {
//       const result = await testDB.pool.request().query('SELECT 1 as test');
//       expect(result.recordset[0].test).toBe(1);
//     });

//     it('should maintain connection state through public pool field', async () => {
//       // MSSQL: Use SYSDATETIME() for current timestamp
//       const result1 = await testDB.pool.request().query('SELECT SYSDATETIME() as timestamp1');
//       const result2 = await testDB.pool.request().query('SELECT SYSDATETIME() as timestamp2');

//       expect(result1.recordset[0].timestamp1).toBeDefined();
//       expect(result2.recordset[0].timestamp2).toBeDefined();
//       // Compare timestamps as strings (ISO format)
//       expect(result2.recordset[0].timestamp2 >= result1.recordset[0].timestamp1).toBe(true);
//     });

//     it('should throw error when pool is used after disconnect', async () => {
//       await testDB.close();
//       await expect(testDB.pool.request().query('SELECT 1')).rejects.toThrow();
//     });
//   });

//   beforeEach(async () => {
//     // Only clear tables if store is initialized
//     try {
//       // Clear tables before each test
//       await store.clearTable({ tableName: TABLE_WORKFLOW_SNAPSHOT });
//       await store.clearTable({ tableName: TABLE_MESSAGES });
//       await store.clearTable({ tableName: TABLE_THREADS });
//       await store.clearTable({ tableName: TABLE_EVALS });
//       await store.clearTable({ tableName: TABLE_TRACES });
//     } catch (error) {
//       // Ignore errors during table clearing
//       console.warn('Error clearing tables:', error);
//     }
//   });

//   // --- Validation tests ---
//   describe('Validation', () => {
//     const validConfig = TEST_CONFIG;
//     it('throws if connectionString is empty', () => {
//       expect(() => new MSSQLStore({ connectionString: '' })).toThrow(
//         /connectionString must be provided and cannot be empty/,
//       );
//     });
//     it('throws if server is missing or empty', () => {
//       expect(() => new MSSQLStore({ ...validConfig, server: '' })).toThrow(
//         /server must be provided and cannot be empty/,
//       );
//       const { server, ...rest } = validConfig;
//       expect(() => new MSSQLStore(rest as any)).toThrow(/server must be provided and cannot be empty/);
//     });
//     it('throws if user is missing or empty', () => {
//       expect(() => new MSSQLStore({ ...validConfig, user: '' })).toThrow(/user must be provided and cannot be empty/);
//       const { user, ...rest } = validConfig;
//       expect(() => new MSSQLStore(rest as any)).toThrow(/user must be provided and cannot be empty/);
//     });
//     it('throws if database is missing or empty', () => {
//       expect(() => new MSSQLStore({ ...validConfig, database: '' })).toThrow(
//         /database must be provided and cannot be empty/,
//       );
//       const { database, ...rest } = validConfig;
//       expect(() => new MSSQLStore(rest as any)).toThrow(/database must be provided and cannot be empty/);
//     });
//     it('throws if password is missing or empty', () => {
//       expect(() => new MSSQLStore({ ...validConfig, password: '' })).toThrow(
//         /password must be provided and cannot be empty/,
//       );
//       const { password, ...rest } = validConfig;
//       expect(() => new MSSQLStore(rest as any)).toThrow(/password must be provided and cannot be empty/);
//     });
//     it('does not throw on valid config (host-based)', () => {
//       expect(() => new MSSQLStore(validConfig)).not.toThrow();
//     });
//     it('does not throw on non-empty connection string', () => {
//       expect(() => new MSSQLStore({ connectionString })).not.toThrow();
//     });
//   });

//   describe('Thread Operations', () => {
//     it('should create and retrieve a thread', async () => {
//       const thread = createSampleThread();

//       // Save thread
//       const savedThread = await store.saveThread({ thread });
//       expect(savedThread).toEqual(thread);

//       // Retrieve thread
//       const retrievedThread = await store.getThreadById({ threadId: thread.id });
//       expect(retrievedThread?.title).toEqual(thread.title);
//     });

//     it('should return null for non-existent thread', async () => {
//       const result = await store.getThreadById({ threadId: 'non-existent' });
//       expect(result).toBeNull();
//     });

//     it('should get threads by resource ID', async () => {
//       const thread1 = createSampleThread();
//       const thread2 = { ...createSampleThread(), resourceId: thread1.resourceId };

//       await store.saveThread({ thread: thread1 });
//       await store.saveThread({ thread: thread2 });

//       const threads = await store.getThreadsByResourceId({ resourceId: thread1.resourceId });
//       expect(threads).toHaveLength(2);
//       expect(threads.map(t => t.id)).toEqual(expect.arrayContaining([thread1.id, thread2.id]));
//     });

//     it('should update thread title and metadata', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       const newMetadata = { newKey: 'newValue' };
//       const updatedThread = await store.updateThread({
//         id: thread.id,
//         title: 'Updated Title',
//         metadata: newMetadata,
//       });

//       expect(updatedThread.title).toBe('Updated Title');
//       expect(updatedThread.metadata).toEqual({
//         ...thread.metadata,
//         ...newMetadata,
//       });

//       // Verify persistence
//       const retrievedThread = await store.getThreadById({ threadId: thread.id });
//       expect(retrievedThread).toEqual(updatedThread);
//     });

//     it('should delete thread and its messages', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       // Add some messages
//       const messages = [createSampleMessageV1({ threadId: thread.id }), createSampleMessageV1({ threadId: thread.id })];
//       await store.saveMessages({ messages });

//       await store.deleteThread({ threadId: thread.id });

//       const retrievedThread = await store.getThreadById({ threadId: thread.id });
//       expect(retrievedThread).toBeNull();

//       // Verify messages were also deleted
//       const retrievedMessages = await store.getMessages({ threadId: thread.id });
//       expect(retrievedMessages).toHaveLength(0);
//     });

//     it('should update thread updatedAt when a message is saved to it', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       // Get the initial thread to capture the original updatedAt
//       const initialThread = await store.getThreadById({ threadId: thread.id });
//       expect(initialThread).toBeDefined();
//       const originalUpdatedAt = initialThread!.updatedAt;

//       // Wait a small amount to ensure different timestamp
//       await new Promise(resolve => setTimeout(resolve, 10));

//       // Create and save a message to the thread
//       const message = createSampleMessageV1({ threadId: thread.id });
//       await store.saveMessages({ messages: [message] });

//       // Retrieve the thread again and check that updatedAt was updated
//       const updatedThread = await store.getThreadById({ threadId: thread.id });
//       expect(updatedThread).toBeDefined();
//       expect(updatedThread!.updatedAt.getTime()).toBeGreaterThan(originalUpdatedAt.getTime());
//     });
//   });

//   describe('Message Operations', () => {
//     it('should save and retrieve messages', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       const messages = [createSampleMessageV1({ threadId: thread.id }), createSampleMessageV1({ threadId: thread.id })];

//       // Save messages
//       const savedMessages = await store.saveMessages({ messages });

//       // Retrieve messages
//       const retrievedMessages = await store.getMessages({ threadId: thread.id, format: 'v1' });

//       const checkMessages = messages.map(m => {
//         const { resourceId, ...rest } = m;
//         return rest;
//       });

//       expect(savedMessages).toEqual(messages);
//       expect(retrievedMessages).toHaveLength(2);
//       expect(retrievedMessages).toEqual(expect.arrayContaining(checkMessages));
//     });

//     it('should handle empty message array', async () => {
//       const result = await store.saveMessages({ messages: [] });
//       expect(result).toEqual([]);
//     });

//     it('should maintain message order', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       const messageContent = ['First', 'Second', 'Third'];

//       const messages = messageContent.map(content =>
//         createSampleMessageV2({ threadId: thread.id, content: { content, parts: [{ type: 'text', text: content }] } }),
//       );

//       await store.saveMessages({ messages, format: 'v2' });

//       const retrievedMessages = await store.getMessages({ threadId: thread.id, format: 'v2' });
//       expect(retrievedMessages).toHaveLength(3);

//       // Verify order is maintained
//       retrievedMessages.forEach((msg, idx) => {
//         expect((msg.content.parts[0] as any).text).toEqual(messageContent[idx]);
//       });
//     });

//     it('should rollback on error during message save', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       const messages = [
//         createSampleMessageV1({ threadId: thread.id }),
//         { ...createSampleMessageV1({ threadId: thread.id }), id: null } as any, // This will cause an error
//       ];

//       await expect(store.saveMessages({ messages })).rejects.toThrow();

//       // Verify no messages were saved
//       const savedMessages = await store.getMessages({ threadId: thread.id });
//       expect(savedMessages).toHaveLength(0);
//     });

//     it('should retrieve messages w/ next/prev messages by message id + resource id', async () => {
//       const thread = createSampleThread({ id: 'thread-one' });
//       await store.saveThread({ thread });

//       const thread2 = createSampleThread({ id: 'thread-two' });
//       await store.saveThread({ thread: thread2 });

//       const thread3 = createSampleThread({ id: 'thread-three' });
//       await store.saveThread({ thread: thread3 });

//       const messages: MastraMessageV2[] = [
//         createSampleMessageV2({
//           threadId: 'thread-one',
//           content: { content: 'First' },
//           resourceId: 'cross-thread-resource',
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-one',
//           content: { content: 'Second' },
//           resourceId: 'cross-thread-resource',
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-one',
//           content: { content: 'Third' },
//           resourceId: 'cross-thread-resource',
//         }),

//         createSampleMessageV2({
//           threadId: 'thread-two',
//           content: { content: 'Fourth' },
//           resourceId: 'cross-thread-resource',
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-two',
//           content: { content: 'Fifth' },
//           resourceId: 'cross-thread-resource',
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-two',
//           content: { content: 'Sixth' },
//           resourceId: 'cross-thread-resource',
//         }),

//         createSampleMessageV2({
//           threadId: 'thread-three',
//           content: { content: 'Seventh' },
//           resourceId: 'other-resource',
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-three',
//           content: { content: 'Eighth' },
//           resourceId: 'other-resource',
//         }),
//       ];

//       await store.saveMessages({ messages: messages, format: 'v2' });

//       const retrievedMessages = await store.getMessages({ threadId: 'thread-one', format: 'v2' });
//       expect(retrievedMessages).toHaveLength(3);
//       expect(retrievedMessages.map((m: any) => m.content.parts[0].text)).toEqual(['First', 'Second', 'Third']);

//       const retrievedMessages2 = await store.getMessages({ threadId: 'thread-two', format: 'v2' });
//       expect(retrievedMessages2).toHaveLength(3);
//       expect(retrievedMessages2.map((m: any) => m.content.parts[0].text)).toEqual(['Fourth', 'Fifth', 'Sixth']);

//       const retrievedMessages3 = await store.getMessages({ threadId: 'thread-three', format: 'v2' });
//       expect(retrievedMessages3).toHaveLength(2);
//       expect(retrievedMessages3.map((m: any) => m.content.parts[0].text)).toEqual(['Seventh', 'Eighth']);

//       const { messages: crossThreadMessages } = await store.getMessagesPaginated({
//         threadId: 'thread-doesnt-exist',
//         format: 'v2',
//         selectBy: {
//           last: 0,
//           include: [
//             {
//               id: messages[1].id,
//               threadId: 'thread-one',
//               withNextMessages: 2,
//               withPreviousMessages: 2,
//             },
//             {
//               id: messages[4].id,
//               threadId: 'thread-two',
//               withPreviousMessages: 2,
//               withNextMessages: 2,
//             },
//           ],
//         },
//       });

//       expect(crossThreadMessages).toHaveLength(6);
//       expect(crossThreadMessages.filter(m => m.threadId === `thread-one`)).toHaveLength(3);
//       expect(crossThreadMessages.filter(m => m.threadId === `thread-two`)).toHaveLength(3);
//     });

//     it('should return messages using both last and include (cross-thread, deduped)', async () => {
//       const thread = createSampleThread({ id: 'thread-one' });
//       await store.saveThread({ thread });

//       const thread2 = createSampleThread({ id: 'thread-two' });
//       await store.saveThread({ thread: thread2 });

//       const now = new Date();

//       // Setup: create messages in two threads
//       const messages = [
//         createSampleMessageV2({
//           threadId: 'thread-one',
//           content: { content: 'A' },
//           createdAt: new Date(now.getTime()),
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-one',
//           content: { content: 'B' },
//           createdAt: new Date(now.getTime() + 1000),
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-one',
//           content: { content: 'C' },
//           createdAt: new Date(now.getTime() + 2000),
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-two',
//           content: { content: 'D' },
//           createdAt: new Date(now.getTime() + 3000),
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-two',
//           content: { content: 'E' },
//           createdAt: new Date(now.getTime() + 4000),
//         }),
//         createSampleMessageV2({
//           threadId: 'thread-two',
//           content: { content: 'F' },
//           createdAt: new Date(now.getTime() + 5000),
//         }),
//       ];
//       await store.saveMessages({ messages, format: 'v2' });

//       // Use last: 2 and include a message from another thread with context
//       const result = await store.getMessages({
//         threadId: 'thread-one',
//         format: 'v2',
//         selectBy: {
//           last: 2,
//           include: [
//             {
//               id: messages[4].id, // 'E' from thread-bar
//               threadId: 'thread-two',
//               withPreviousMessages: 1,
//               withNextMessages: 1,
//             },
//           ],
//         },
//       });

//       // Should include last 2 from thread-one and 3 from thread-two (D, E, F)
//       expect(result.map(m => m.content.content).sort()).toEqual(['B', 'C', 'D', 'E', 'F']);
//       // Should include 2 from thread-one
//       expect(result.filter(m => m.threadId === 'thread-one').map(m => m.content.content)).toEqual(['B', 'C']);
//       // Should include 3 from thread-two
//       expect(result.filter(m => m.threadId === 'thread-two').map(m => m.content.content)).toEqual(['D', 'E', 'F']);
//     });
//   });

//   describe('updateMessages', () => {
//     let thread: StorageThreadType;

//     beforeEach(async () => {
//       const threadData = createSampleThread();
//       thread = await store.saveThread({ thread: threadData as StorageThreadType });
//     });

//     it('should update a single field of a message (e.g., role)', async () => {
//       const originalMessage = createSampleMessageV2({ threadId: thread.id, role: 'user', thread });
//       await store.saveMessages({ messages: [originalMessage], format: 'v2' });

//       const updatedMessages = await store.updateMessages({
//         messages: [{ id: originalMessage.id, role: 'assistant' }],
//       });

//       expect(updatedMessages).toHaveLength(1);
//       expect(updatedMessages[0].role).toBe('assistant');
//       expect(updatedMessages[0].content).toEqual(originalMessage.content); // Ensure content is unchanged
//     });

//     it('should update only the metadata within the content field, preserving other content', async () => {
//       const originalMessage = createSampleMessageV2({
//         threadId: thread.id,
//         content: { content: 'hello world', parts: [{ type: 'text', text: 'hello world' }] },
//         thread,
//       });
//       await store.saveMessages({ messages: [originalMessage], format: 'v2' });

//       const newMetadata = { someKey: 'someValue' };
//       await store.updateMessages({
//         messages: [{ id: originalMessage.id, content: { metadata: newMetadata } as any }],
//       });

//       const fromDb = await store.getMessages({ threadId: thread.id, format: 'v2' });
//       expect(fromDb[0].content.metadata).toEqual(newMetadata);
//       expect(fromDb[0].content.content).toBe('hello world');
//       expect(fromDb[0].content.parts).toEqual([{ type: 'text', text: 'hello world' }]);
//     });

//     it('should deep merge metadata, not overwrite it', async () => {
//       const originalMessage = createSampleMessageV2({
//         threadId: thread.id,
//         content: { metadata: { initial: true }, content: 'old content' },
//         thread,
//       });
//       await store.saveMessages({ messages: [originalMessage], format: 'v2' });

//       const newMetadata = { updated: true };
//       await store.updateMessages({
//         messages: [{ id: originalMessage.id, content: { metadata: newMetadata } as any }],
//       });

//       const fromDb = await store.getMessages({ threadId: thread.id, format: 'v2' });
//       expect(fromDb[0].content.metadata).toEqual({ initial: true, updated: true });
//     });

//     it('should update multiple messages at once', async () => {
//       const msg1 = createSampleMessageV2({ threadId: thread.id, role: 'user', thread });
//       const msg2 = createSampleMessageV2({ threadId: thread.id, content: { content: 'original' }, thread });
//       await store.saveMessages({ messages: [msg1, msg2], format: 'v2' });

//       await store.updateMessages({
//         messages: [
//           { id: msg1.id, role: 'assistant' },
//           { id: msg2.id, content: { content: 'updated' } as any },
//         ],
//       });

//       const fromDb = await store.getMessages({ threadId: thread.id, format: 'v2' });
//       const updatedMsg1 = fromDb.find(m => m.id === msg1.id)!;
//       const updatedMsg2 = fromDb.find(m => m.id === msg2.id)!;

//       expect(updatedMsg1.role).toBe('assistant');
//       expect(updatedMsg2.content.content).toBe('updated');
//     });

//     it('should update the parent thread updatedAt timestamp', async () => {
//       const originalMessage = createSampleMessageV2({ threadId: thread.id, thread });
//       await store.saveMessages({ messages: [originalMessage], format: 'v2' });
//       const initialThread = await store.getThreadById({ threadId: thread.id });

//       await new Promise(r => setTimeout(r, 10));

//       await store.updateMessages({ messages: [{ id: originalMessage.id, role: 'assistant' }] });

//       const updatedThread = await store.getThreadById({ threadId: thread.id });

//       expect(new Date(updatedThread!.updatedAt).getTime()).toBeGreaterThan(
//         new Date(initialThread!.updatedAt).getTime(),
//       );
//     });

//     it('should update timestamps on both threads when moving a message', async () => {
//       const thread2 = await store.saveThread({ thread: createSampleThread() });
//       const message = createSampleMessageV2({ threadId: thread.id, thread });
//       await store.saveMessages({ messages: [message], format: 'v2' });

//       const initialThread1 = await store.getThreadById({ threadId: thread.id });
//       const initialThread2 = await store.getThreadById({ threadId: thread2.id });

//       await new Promise(r => setTimeout(r, 10));

//       await store.updateMessages({
//         messages: [{ id: message.id, threadId: thread2.id }],
//       });

//       const updatedThread1 = await store.getThreadById({ threadId: thread.id });
//       const updatedThread2 = await store.getThreadById({ threadId: thread2.id });

//       expect(new Date(updatedThread1!.updatedAt).getTime()).toBeGreaterThan(
//         new Date(initialThread1!.updatedAt).getTime(),
//       );
//       expect(new Date(updatedThread2!.updatedAt).getTime()).toBeGreaterThan(
//         new Date(initialThread2!.updatedAt).getTime(),
//       );

//       // Verify the message was moved
//       const thread1Messages = await store.getMessages({ threadId: thread.id, format: 'v2' });
//       const thread2Messages = await store.getMessages({ threadId: thread2.id, format: 'v2' });
//       expect(thread1Messages).toHaveLength(0);
//       expect(thread2Messages).toHaveLength(1);
//       expect(thread2Messages[0].id).toBe(message.id);
//     });
//     it('should upsert messages: duplicate id+threadId results in update, not duplicate row', async () => {
//       const thread = await createSampleThread();
//       await store.saveThread({ thread });
//       const baseMessage = createSampleMessageV2({
//         threadId: thread.id,
//         createdAt: new Date(),
//         content: { content: 'Original' },
//         resourceId: thread.resourceId,
//       });

//       // Insert the message for the first time
//       await store.saveMessages({ messages: [baseMessage], format: 'v2' });

//       // Insert again with the same id and threadId but different content
//       const updatedMessage = {
//         ...createSampleMessageV2({
//           threadId: thread.id,
//           createdAt: new Date(),
//           content: { content: 'Updated' },
//           resourceId: thread.resourceId,
//         }),
//         id: baseMessage.id,
//       };

//       await store.saveMessages({ messages: [updatedMessage], format: 'v2' });

//       // Retrieve messages for the thread
//       const retrievedMessages = await store.getMessages({ threadId: thread.id, format: 'v2' });

//       // Only one message should exist for that id+threadId
//       expect(retrievedMessages.filter(m => m.id === baseMessage.id)).toHaveLength(1);

//       // The content should be the updated one
//       expect(retrievedMessages.find(m => m.id === baseMessage.id)?.content.content).toBe('Updated');
//     });

//     it('should upsert messages: duplicate id and different threadid', async () => {
//       const thread1 = await createSampleThread();
//       const thread2 = await createSampleThread();
//       await store.saveThread({ thread: thread1 });
//       await store.saveThread({ thread: thread2 });

//       const message = createSampleMessageV2({
//         threadId: thread1.id,
//         createdAt: new Date(),
//         content: { content: 'Thread1 Content' },
//         resourceId: thread1.resourceId,
//       });

//       // Insert message into thread1
//       await store.saveMessages({ messages: [message], format: 'v2' });

//       // Attempt to insert a message with the same id but different threadId
//       const conflictingMessage = {
//         ...createSampleMessageV2({
//           threadId: thread2.id, // different thread
//           content: { content: 'Thread2 Content' },
//           resourceId: thread2.resourceId,
//         }),
//         id: message.id,
//       };

//       // Save should move the message to the new thread
//       await store.saveMessages({ messages: [conflictingMessage], format: 'v2' });

//       // Retrieve messages for both threads
//       const thread1Messages = await store.getMessages({ threadId: thread1.id, format: 'v2' });
//       const thread2Messages = await store.getMessages({ threadId: thread2.id, format: 'v2' });

//       // Thread 1 should NOT have the message with that id
//       expect(thread1Messages.find(m => m.id === message.id)).toBeUndefined();

//       // Thread 2 should have the message with that id
//       expect(thread2Messages.find(m => m.id === message.id)?.content.content).toBe('Thread2 Content');
//     });
//   });

//   describe('Edge Cases and Error Handling', () => {
//     it('should handle large metadata objects', async () => {
//       const thread = createSampleThread();
//       const largeMetadata = {
//         ...thread.metadata,
//         largeArray: Array.from({ length: 1000 }, (_, i) => ({ index: i, data: 'test'.repeat(100) })),
//       };

//       const threadWithLargeMetadata = {
//         ...thread,
//         metadata: largeMetadata,
//       };

//       await store.saveThread({ thread: threadWithLargeMetadata });
//       const retrieved = await store.getThreadById({ threadId: thread.id });

//       expect(retrieved?.metadata).toEqual(largeMetadata);
//     });

//     it('should handle special characters in thread titles', async () => {
//       const thread = {
//         ...createSampleThread(),
//         title: 'Special \'quotes\' and "double quotes" and emoji 🎉',
//       };

//       await store.saveThread({ thread });
//       const retrieved = await store.getThreadById({ threadId: thread.id });

//       expect(retrieved?.title).toBe(thread.title);
//     });

//     it('should handle concurrent thread updates', async () => {
//       const thread = createSampleThread();
//       await store.saveThread({ thread });

//       // Perform multiple updates concurrently
//       const updates = Array.from({ length: 5 }, (_, i) =>
//         store.updateThread({
//           id: thread.id,
//           title: `Update ${i}`,
//           metadata: { update: i },
//         }),
//       );

//       await expect(Promise.all(updates)).resolves.toBeDefined();

//       // Verify final state
//       const finalThread = await store.getThreadById({ threadId: thread.id });
//       expect(finalThread).toBeDefined();
//     });
//   });

//   describe('Workflow Snapshots', () => {
//     it('should persist and load workflow snapshots', async () => {
//       const workflowName = 'test-workflow';
//       const runId = `run-${randomUUID()}`;
//       const snapshot = {
//         status: 'running',
//         context: {
//           input: { type: 'manual' },
//           step1: { status: 'success', output: { data: 'test' } },
//         },
//         value: {},
//         activePaths: [],
//         suspendedPaths: {},
//         runId,
//         timestamp: new Date().getTime(),
//         serializedStepGraph: [],
//       } as unknown as WorkflowRunState;

//       await store.persistWorkflowSnapshot({
//         workflowName,
//         runId,
//         snapshot,
//       });

//       const loadedSnapshot = await store.loadWorkflowSnapshot({
//         workflowName,
//         runId,
//       });

//       expect(loadedSnapshot).toEqual(snapshot);
//     });

//     it('should return null for non-existent workflow snapshot', async () => {
//       const result = await store.loadWorkflowSnapshot({
//         workflowName: 'non-existent',
//         runId: 'non-existent',
//       });

//       expect(result).toBeNull();
//     });

//     it('should update existing workflow snapshot', async () => {
//       const workflowName = 'test-workflow';
//       const runId = `run-${randomUUID()}`;
//       const initialSnapshot = {
//         status: 'running',
//         context: {
//           input: { type: 'manual' },
//         },
//         value: {},
//         activePaths: [],
//         suspendedPaths: {},
//         runId,
//         timestamp: new Date().getTime(),
//         serializedStepGraph: [],
//       };

//       await store.persistWorkflowSnapshot({
//         workflowName,
//         runId,
//         snapshot: initialSnapshot as unknown as WorkflowRunState,
//       });

//       const updatedSnapshot = {
//         status: 'success',
//         context: {
//           input: { type: 'manual' },
//           'step-1': { status: 'success', result: { data: 'test' } },
//         },
//         value: {},
//         activePaths: [],
//         suspendedPaths: {},
//         runId,
//         timestamp: new Date().getTime(),
//       };

//       await store.persistWorkflowSnapshot({
//         workflowName,
//         runId,
//         snapshot: updatedSnapshot as unknown as WorkflowRunState,
//       });

//       const loadedSnapshot = await store.loadWorkflowSnapshot({
//         workflowName,
//         runId,
//       });

//       expect(loadedSnapshot).toEqual(updatedSnapshot);
//     });

//     it('should handle complex workflow state', async () => {
//       const workflowName = 'complex-workflow';
//       const runId = `run-${randomUUID()}`;
//       const complexSnapshot = {
//         value: { currentState: 'running' },
//         context: {
//           'step-1': {
//             status: 'success',
//             output: {
//               nestedData: {
//                 array: [1, 2, 3],
//                 object: { key: 'value' },
//                 date: new Date().toISOString(),
//               },
//             },
//           },
//           'step-2': {
//             status: 'waiting',
//             dependencies: ['step-3', 'step-4'],
//           },
//           input: {
//             type: 'scheduled',
//             metadata: {
//               schedule: '0 0 * * *',
//               timezone: 'UTC',
//             },
//           },
//         },
//         activePaths: [
//           {
//             stepPath: ['step-1'],
//             stepId: 'step-1',
//             status: 'success',
//           },
//           {
//             stepPath: ['step-2'],
//             stepId: 'step-2',
//             status: 'waiting',
//           },
//         ],
//         suspendedPaths: {},
//         runId: runId,
//         timestamp: Date.now(),
//         serializedStepGraph: [],
//         status: 'running',
//       };

//       await store.persistWorkflowSnapshot({
//         workflowName,
//         runId,
//         snapshot: complexSnapshot as unknown as WorkflowRunState,
//       });

//       const loadedSnapshot = await store.loadWorkflowSnapshot({
//         workflowName,
//         runId,
//       });

//       expect(loadedSnapshot).toEqual(complexSnapshot);
//     });
//   });

//   describe('getWorkflowRuns', () => {
//     beforeEach(async () => {
//       await store.clearTable({ tableName: TABLE_WORKFLOW_SNAPSHOT });
//     });
//     it('returns empty array when no workflows exist', async () => {
//       const { runs, total } = await store.getWorkflowRuns();
//       expect(runs).toEqual([]);
//       expect(total).toBe(0);
//     });

//     it('returns all workflows by default', async () => {
//       const workflowName1 = 'default_test_1';
//       const workflowName2 = 'default_test_2';

//       const { snapshot: workflow1, runId: runId1, stepId: stepId1 } = createSampleWorkflowSnapshot('success');
//       const { snapshot: workflow2, runId: runId2, stepId: stepId2 } = createSampleWorkflowSnapshot('failed');

//       await store.persistWorkflowSnapshot({ workflowName: workflowName1, runId: runId1, snapshot: workflow1 });
//       await new Promise(resolve => setTimeout(resolve, 10)); // Small delay to ensure different timestamps
//       await store.persistWorkflowSnapshot({ workflowName: workflowName2, runId: runId2, snapshot: workflow2 });

//       const { runs, total } = await store.getWorkflowRuns();
//       expect(runs).toHaveLength(2);
//       expect(total).toBe(2);
//       expect(runs[0]!.workflowName).toBe(workflowName2); // Most recent first
//       expect(runs[1]!.workflowName).toBe(workflowName1);
//       const firstSnapshot = runs[0]!.snapshot;
//       const secondSnapshot = runs[1]!.snapshot;
//       checkWorkflowSnapshot(firstSnapshot, stepId2, 'failed');
//       checkWorkflowSnapshot(secondSnapshot, stepId1, 'success');
//     });

//     it('filters by workflow name', async () => {
//       const workflowName1 = 'filter_test_1';
//       const workflowName2 = 'filter_test_2';

//       const { snapshot: workflow1, runId: runId1, stepId: stepId1 } = createSampleWorkflowSnapshot('success');
//       const { snapshot: workflow2, runId: runId2 } = createSampleWorkflowSnapshot('failed');

//       await store.persistWorkflowSnapshot({ workflowName: workflowName1, runId: runId1, snapshot: workflow1 });
//       await new Promise(resolve => setTimeout(resolve, 10)); // Small delay to ensure different timestamps
//       await store.persistWorkflowSnapshot({ workflowName: workflowName2, runId: runId2, snapshot: workflow2 });

//       const { runs, total } = await store.getWorkflowRuns({ workflowName: workflowName1 });
//       expect(runs).toHaveLength(1);
//       expect(total).toBe(1);
//       expect(runs[0]!.workflowName).toBe(workflowName1);
//       const snapshot = runs[0]!.snapshot;
//       checkWorkflowSnapshot(snapshot, stepId1, 'success');
//     });

//     it('filters by date range', async () => {
//       const now = new Date();
//       const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
//       const twoDaysAgo = new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000);
//       const workflowName1 = 'date_test_1';
//       const workflowName2 = 'date_test_2';
//       const workflowName3 = 'date_test_3';

//       const { snapshot: workflow1, runId: runId1 } = createSampleWorkflowSnapshot('success');
//       const { snapshot: workflow2, runId: runId2, stepId: stepId2 } = createSampleWorkflowSnapshot('failed');
//       const { snapshot: workflow3, runId: runId3, stepId: stepId3 } = createSampleWorkflowSnapshot('suspended');

//       await store.insert({
//         tableName: TABLE_WORKFLOW_SNAPSHOT,
//         record: {
//           workflow_name: workflowName1,
//           run_id: runId1,
//           snapshot: workflow1,
//           createdAt: twoDaysAgo,
//           updatedAt: twoDaysAgo,
//         },
//       });

//       await store.insert({
//         tableName: TABLE_WORKFLOW_SNAPSHOT,
//         record: {
//           workflow_name: workflowName2,
//           run_id: runId2,
//           snapshot: workflow2,
//           createdAt: yesterday,
//           updatedAt: yesterday,
//         },
//       });
//       await store.insert({
//         tableName: TABLE_WORKFLOW_SNAPSHOT,
//         record: {
//           workflow_name: workflowName3,
//           run_id: runId3,
//           snapshot: workflow3,
//           createdAt: now,
//           updatedAt: now,
//         },
//       });

//       const { runs } = await store.getWorkflowRuns({
//         fromDate: yesterday,
//         toDate: now,
//       });

//       expect(runs).toHaveLength(2);
//       expect(runs[0]!.workflowName).toBe(workflowName3);
//       expect(runs[1]!.workflowName).toBe(workflowName2);
//       const firstSnapshot = runs[0]!.snapshot;
//       const secondSnapshot = runs[1]!.snapshot;
//       checkWorkflowSnapshot(firstSnapshot, stepId3, 'suspended');
//       checkWorkflowSnapshot(secondSnapshot, stepId2, 'failed');
//     });

//     it('handles pagination', async () => {
//       const workflowName1 = 'page_test_1';
//       const workflowName2 = 'page_test_2';
//       const workflowName3 = 'page_test_3';

//       const { snapshot: workflow1, runId: runId1, stepId: stepId1 } = createSampleWorkflowSnapshot('success');
//       const { snapshot: workflow2, runId: runId2, stepId: stepId2 } = createSampleWorkflowSnapshot('failed');
//       const { snapshot: workflow3, runId: runId3, stepId: stepId3 } = createSampleWorkflowSnapshot('suspended');

//       await store.persistWorkflowSnapshot({ workflowName: workflowName1, runId: runId1, snapshot: workflow1 });
//       await new Promise(resolve => setTimeout(resolve, 10)); // Small delay to ensure different timestamps
//       await store.persistWorkflowSnapshot({ workflowName: workflowName2, runId: runId2, snapshot: workflow2 });
//       await new Promise(resolve => setTimeout(resolve, 10)); // Small delay to ensure different timestamps
//       await store.persistWorkflowSnapshot({ workflowName: workflowName3, runId: runId3, snapshot: workflow3 });

//       // Get first page
//       const page1 = await store.getWorkflowRuns({ limit: 2, offset: 0 });
//       expect(page1.runs).toHaveLength(2);
//       expect(page1.total).toBe(3); // Total count of all records
//       expect(page1.runs[0]!.workflowName).toBe(workflowName3);
//       expect(page1.runs[1]!.workflowName).toBe(workflowName2);
//       const firstSnapshot = page1.runs[0]!.snapshot;
//       const secondSnapshot = page1.runs[1]!.snapshot;
//       checkWorkflowSnapshot(firstSnapshot, stepId3, 'suspended');
//       checkWorkflowSnapshot(secondSnapshot, stepId2, 'failed');

//       // Get second page
//       const page2 = await store.getWorkflowRuns({ limit: 2, offset: 2 });
//       expect(page2.runs).toHaveLength(1);
//       expect(page2.total).toBe(3);
//       expect(page2.runs[0]!.workflowName).toBe(workflowName1);
//       const snapshot = page2.runs[0]!.snapshot;
//       checkWorkflowSnapshot(snapshot, stepId1, 'success');
//     });
//   });

//   describe('getWorkflowRunById', () => {
//     const workflowName = 'workflow-id-test';
//     let runId: string;
//     let stepId: string;

//     beforeEach(async () => {
//       // Insert a workflow run for positive test
//       const sample = createSampleWorkflowSnapshot('success');
//       runId = sample.runId;
//       stepId = sample.stepId;
//       await store.insert({
//         tableName: TABLE_WORKFLOW_SNAPSHOT,
//         record: {
//           workflow_name: workflowName,
//           run_id: runId,
//           resourceId: 'resource-abc',
//           snapshot: sample.snapshot,
//           createdAt: new Date(),
//           updatedAt: new Date(),
//         },
//       });
//     });

//     it('should retrieve a workflow run by ID', async () => {
//       const found = await store.getWorkflowRunById({
//         runId,
//         workflowName,
//       });
//       expect(found).not.toBeNull();
//       expect(found?.runId).toBe(runId);
//       checkWorkflowSnapshot(found?.snapshot!, stepId, 'success');
//     });

//     it('should return null for non-existent workflow run ID', async () => {
//       const notFound = await store.getWorkflowRunById({
//         runId: 'non-existent-id',
//         workflowName,
//       });
//       expect(notFound).toBeNull();
//     });
//   });
//   describe('getWorkflowRuns with resourceId', () => {
//     const workflowName = 'workflow-id-test';
//     let resourceId: string;
//     let runIds: string[] = [];

//     beforeEach(async () => {
//       // Insert multiple workflow runs for the same resourceId
//       resourceId = 'resource-shared';
//       for (const status of ['success', 'failed']) {
//         const sample = createSampleWorkflowSnapshot(status as WorkflowRunState['context'][string]['status']);
//         runIds.push(sample.runId);
//         await store.insert({
//           tableName: TABLE_WORKFLOW_SNAPSHOT,
//           record: {
//             workflow_name: workflowName,
//             run_id: sample.runId,
//             resourceId,
//             snapshot: sample.snapshot,
//             createdAt: new Date(),
//             updatedAt: new Date(),
//           },
//         });
//       }
//       // Insert a run with a different resourceId
//       const other = createSampleWorkflowSnapshot('suspended');
//       await store.insert({
//         tableName: TABLE_WORKFLOW_SNAPSHOT,
//         record: {
//           workflow_name: workflowName,
//           run_id: other.runId,
//           resourceId: 'resource-other',
//           snapshot: other.snapshot,
//           createdAt: new Date(),
//           updatedAt: new Date(),
//         },
//       });
//     });

//     it('should retrieve all workflow runs by resourceId', async () => {
//       const { runs } = await store.getWorkflowRuns({
//         resourceId,
//         workflowName,
//       });
//       expect(Array.isArray(runs)).toBe(true);
//       expect(runs.length).toBeGreaterThanOrEqual(2);
//       for (const run of runs) {
//         expect(run.resourceId).toBe(resourceId);
//       }
//     });

//     it('should return an empty array if no workflow runs match resourceId', async () => {
//       const { runs } = await store.getWorkflowRuns({
//         resourceId: 'non-existent-resource',
//         workflowName,
//       });
//       expect(Array.isArray(runs)).toBe(true);
//       expect(runs.length).toBe(0);
//     });
//   });

//   describe('Eval Operations', () => {
//     it('should retrieve evals by agent name', async () => {
//       const agentName = `test-agent-${randomUUID()}`;

//       // Create sample evals using the imported helper
//       const liveEval = createSampleEval(agentName, false); // createSampleEval returns snake_case
//       const testEval = createSampleEval(agentName, true);
//       const otherAgentEval = createSampleEval(`other-agent-${randomUUID()}`, false);

//       // Insert evals - ensure DB columns are snake_case
//       await store.insert({
//         tableName: TABLE_EVALS,
//         record: {
//           agent_name: liveEval.agent_name, // Use snake_case
//           input: liveEval.input,
//           output: liveEval.output,
//           result: liveEval.result,
//           metric_name: liveEval.metric_name, // Use snake_case
//           instructions: liveEval.instructions,
//           test_info: liveEval.test_info, // test_info from helper can be undefined or object
//           global_run_id: liveEval.global_run_id, // Use snake_case
//           run_id: liveEval.run_id, // Use snake_case
//           created_at: new Date(liveEval.created_at as string), // created_at from helper is string or Date
//         },
//       });

//       await store.insert({
//         tableName: TABLE_EVALS,
//         record: {
//           agent_name: testEval.agent_name,
//           input: testEval.input,
//           output: testEval.output,
//           result: testEval.result,
//           metric_name: testEval.metric_name,
//           instructions: testEval.instructions,
//           test_info: testEval.test_info ? JSON.stringify(testEval.test_info) : null,
//           global_run_id: testEval.global_run_id,
//           run_id: testEval.run_id,
//           created_at: new Date(testEval.created_at as string),
//         },
//       });

//       await store.insert({
//         tableName: TABLE_EVALS,
//         record: {
//           agent_name: otherAgentEval.agent_name,
//           input: otherAgentEval.input,
//           output: otherAgentEval.output,
//           result: otherAgentEval.result,
//           metric_name: otherAgentEval.metric_name,
//           instructions: otherAgentEval.instructions,
//           test_info: otherAgentEval.test_info, // Can be null/undefined directly
//           global_run_id: otherAgentEval.global_run_id,
//           run_id: otherAgentEval.run_id,
//           created_at: new Date(otherAgentEval.created_at as string),
//         },
//       });

//       // Test getting all evals for the agent
//       const allEvals = await store.getEvalsByAgentName(agentName);
//       expect(allEvals).toHaveLength(2);
//       // EvalRow type expects camelCase, but MSSQLStore.transformEvalRow converts snake_case from DB to camelCase
//       expect(allEvals.map(e => e.runId)).toEqual(expect.arrayContaining([liveEval.run_id, testEval.run_id]));

//       // Test getting only live evals
//       const liveEvals = await store.getEvalsByAgentName(agentName, 'live');
//       expect(liveEvals).toHaveLength(1);
//       expect(liveEvals[0].runId).toBe(liveEval.run_id); // Comparing with snake_case run_id from original data

//       // Test getting only test evals
//       const testEvalsResult = await store.getEvalsByAgentName(agentName, 'test');
//       expect(testEvalsResult).toHaveLength(1);
//       expect(testEvalsResult[0].runId).toBe(testEval.run_id);
//       expect(testEvalsResult[0].testInfo).toEqual(testEval.test_info);

//       // Test getting evals for non-existent agent
//       const nonExistentEvals = await store.getEvalsByAgentName('non-existent-agent');
//       expect(nonExistentEvals).toHaveLength(0);
//     });
//   });

//   describe('hasColumn', () => {
//     const tempTable = 'temp_test_table';

//     beforeEach(async () => {
//       // Always try to drop the table before each test, ignore errors if it doesn't exist
//       try {
//         await store.pool.query(`DROP TABLE IF EXISTS ${tempTable}`);
//       } catch {
//         /* ignore */
//       }
//     });

//     it('returns true if the column exists', async () => {
//       await store.pool.query(`CREATE TABLE ${tempTable} (id INT IDENTITY(1,1) PRIMARY KEY, resourceId NVARCHAR(MAX))`);
//       expect(await store['hasColumn'](tempTable, 'resourceId')).toBe(true);
//     });

//     it('returns false if the column does not exist', async () => {
//       await store.pool.query(`CREATE TABLE ${tempTable} (id INT IDENTITY(1,1) PRIMARY KEY)`);
//       expect(await store['hasColumn'](tempTable, 'resourceId')).toBe(false);
//     });

//     afterEach(async () => {
//       // Always try to drop the table after each test, ignore errors if it doesn't exist
//       try {
//         await store.pool.query(`DROP TABLE IF EXISTS ${tempTable}`);
//       } catch {
//         /* ignore */
//       }
//     });
//   });

//   describe('alterTable', () => {
//     const TEST_TABLE = 'test_alter_table';
//     const BASE_SCHEMA = {
//       id: { type: 'integer', primaryKey: true, nullable: false },
//       name: { type: 'text', nullable: true },
//     } as Record<string, StorageColumn>;

//     beforeEach(async () => {
//       await store.createTable({ tableName: TEST_TABLE as TABLE_NAMES, schema: BASE_SCHEMA });
//     });

//     afterEach(async () => {
//       await store.clearTable({ tableName: TEST_TABLE as TABLE_NAMES });
//     });

//     it('adds a new column to an existing table', async () => {
//       await store.alterTable({
//         tableName: TEST_TABLE as TABLE_NAMES,
//         schema: { ...BASE_SCHEMA, age: { type: 'integer', nullable: true } },
//         ifNotExists: ['age'],
//       });

//       await store.insert({
//         tableName: TEST_TABLE as TABLE_NAMES,
//         record: { id: 1, name: 'Alice', age: 42 },
//       });

//       const row = await store.load<{ id: string; name: string; age?: number }>({
//         tableName: TEST_TABLE as TABLE_NAMES,
//         keys: { id: '1' },
//       });
//       expect(row?.age).toBe(42);
//     });

//     it('is idempotent when adding an existing column', async () => {
//       await store.alterTable({
//         tableName: TEST_TABLE as TABLE_NAMES,
//         schema: { ...BASE_SCHEMA, foo: { type: 'text', nullable: true } },
//         ifNotExists: ['foo'],
//       });
//       // Add the column again (should not throw)
//       await expect(
//         store.alterTable({
//           tableName: TEST_TABLE as TABLE_NAMES,
//           schema: { ...BASE_SCHEMA, foo: { type: 'text', nullable: true } },
//           ifNotExists: ['foo'],
//         }),
//       ).resolves.not.toThrow();
//     });

//     it('should add a default value to a column when using not null', async () => {
//       await store.insert({
//         tableName: TEST_TABLE as TABLE_NAMES,
//         record: { id: 1, name: 'Bob' },
//       });

//       await expect(
//         store.alterTable({
//           tableName: TEST_TABLE as TABLE_NAMES,
//           schema: { ...BASE_SCHEMA, text_column: { type: 'text', nullable: false } },
//           ifNotExists: ['text_column'],
//         }),
//       ).resolves.not.toThrow();

//       await expect(
//         store.alterTable({
//           tableName: TEST_TABLE as TABLE_NAMES,
//           schema: { ...BASE_SCHEMA, timestamp_column: { type: 'timestamp', nullable: false } },
//           ifNotExists: ['timestamp_column'],
//         }),
//       ).resolves.not.toThrow();

//       await expect(
//         store.alterTable({
//           tableName: TEST_TABLE as TABLE_NAMES,
//           schema: { ...BASE_SCHEMA, bigint_column: { type: 'bigint', nullable: false } },
//           ifNotExists: ['bigint_column'],
//         }),
//       ).resolves.not.toThrow();

//       await expect(
//         store.alterTable({
//           tableName: TEST_TABLE as TABLE_NAMES,
//           schema: { ...BASE_SCHEMA, jsonb_column: { type: 'jsonb', nullable: false } },
//           ifNotExists: ['jsonb_column'],
//         }),
//       ).resolves.not.toThrow();
//     });
//   });

//   describe('Schema Support', () => {
//     const customSchema = 'mastraTest';
//     let customSchemaStore: MSSQLStore;

//     beforeAll(async () => {
//       customSchemaStore = new MSSQLStore({
//         ...TEST_CONFIG,
//         schemaName: customSchema,
//       });

//       await customSchemaStore.init();
//     });

//     afterAll(async () => {
//       await customSchemaStore.close();
//       // Re-initialize the main store for subsequent tests
//       store = new MSSQLStore(TEST_CONFIG);
//       await store.init();
//     });

//     describe('Constructor and Initialization', () => {
//       it('should accept connectionString directly', () => {
//         // Use existing store instead of creating new one
//         expect(store).toBeInstanceOf(MSSQLStore);
//       });

//       it('should accept config object with schema', () => {
//         // Use existing custom schema store
//         expect(customSchemaStore).toBeInstanceOf(MSSQLStore);
//       });
//     });

//     describe('Schema Operations', () => {
//       it('should create and query tables in custom schema', async () => {
//         // Create thread in custom schema
//         const thread = createSampleThread();
//         await customSchemaStore.saveThread({ thread });

//         // Verify thread exists in custom schema
//         const retrieved = await customSchemaStore.getThreadById({ threadId: thread.id });
//         expect(retrieved?.title).toBe(thread.title);
//       });

//       it('should allow same table names in different schemas', async () => {
//         // Create threads in both schemas
//         const defaultThread = createSampleThread();
//         const customThread = createSampleThread();

//         await store.saveThread({ thread: defaultThread });
//         await customSchemaStore.saveThread({ thread: customThread });

//         // Verify threads exist in respective schemas
//         const defaultResult = await store.getThreadById({ threadId: defaultThread.id });
//         const customResult = await customSchemaStore.getThreadById({ threadId: customThread.id });

//         expect(defaultResult?.id).toBe(defaultThread.id);
//         expect(customResult?.id).toBe(customThread.id);

//         // Verify cross-schema isolation
//         const defaultInCustom = await customSchemaStore.getThreadById({ threadId: defaultThread.id });
//         const customInDefault = await store.getThreadById({ threadId: customThread.id });

//         expect(defaultInCustom).toBeNull();
//         expect(customInDefault).toBeNull();
//       });
//     });
//   });

//   describe('Pagination Features', () => {
//     beforeEach(async () => {
//       await store.clearTable({ tableName: TABLE_EVALS });
//       await store.clearTable({ tableName: TABLE_TRACES });
//       await store.clearTable({ tableName: TABLE_MESSAGES });
//       await store.clearTable({ tableName: TABLE_THREADS });
//     });

//     describe('getEvals with pagination', () => {
//       it('should return paginated evals with total count (page/perPage)', async () => {
//         const agentName = 'pagination-agent-evals';
//         const evalPromises = Array.from({ length: 25 }, (_, i) => {
//           const evalData = createSampleEval(agentName, i % 2 === 0);
//           return store.insert({
//             tableName: TABLE_EVALS,
//             record: {
//               run_id: evalData.run_id,
//               agent_name: evalData.agent_name,
//               input: evalData.input,
//               output: evalData.output,
//               result: evalData.result,
//               metric_name: evalData.metric_name,
//               instructions: evalData.instructions,
//               test_info: evalData.test_info,
//               global_run_id: evalData.global_run_id,
//               created_at: new Date(evalData.created_at as string),
//             },
//           });
//         });
//         await Promise.all(evalPromises);

//         const page1 = await store.getEvals({ agentName, page: 0, perPage: 10 });
//         expect(page1.evals).toHaveLength(10);
//         expect(page1.total).toBe(25);
//         expect(page1.page).toBe(0);
//         expect(page1.perPage).toBe(10);
//         expect(page1.hasMore).toBe(true);

//         const page3 = await store.getEvals({ agentName, page: 2, perPage: 10 });
//         expect(page3.evals).toHaveLength(5);
//         expect(page3.total).toBe(25);
//         expect(page3.page).toBe(2);
//         expect(page3.hasMore).toBe(false);
//       });

//       it('should support limit/offset pagination for getEvals', async () => {
//         const agentName = 'pagination-agent-lo-evals';
//         const evalPromises = Array.from({ length: 15 }, () => {
//           const evalData = createSampleEval(agentName);
//           return store.insert({
//             tableName: TABLE_EVALS,
//             record: {
//               run_id: evalData.run_id,
//               agent_name: evalData.agent_name,
//               input: evalData.input,
//               output: evalData.output,
//               result: evalData.result,
//               metric_name: evalData.metric_name,
//               instructions: evalData.instructions,
//               test_info: evalData.test_info,
//               global_run_id: evalData.global_run_id,
//               created_at: new Date(evalData.created_at as string),
//             },
//           });
//         });
//         await Promise.all(evalPromises);

//         const result = await store.getEvals({ agentName, perPage: 5, page: 2 });
//         expect(result.evals).toHaveLength(5);
//         expect(result.total).toBe(15);
//         expect(result.page).toBe(2);
//         expect(result.perPage).toBe(5);
//         expect(result.hasMore).toBe(false);
//       });

//       it('should filter by type with pagination for getEvals', async () => {
//         const agentName = 'pagination-agent-type-evals';
//         const testEvalPromises = Array.from({ length: 10 }, () => {
//           const evalData = createSampleEval(agentName, true);
//           return store.insert({
//             tableName: TABLE_EVALS,
//             record: {
//               run_id: evalData.run_id,
//               agent_name: evalData.agent_name,
//               input: evalData.input,
//               output: evalData.output,
//               result: evalData.result,
//               metric_name: evalData.metric_name,
//               instructions: evalData.instructions,
//               test_info: evalData.test_info,
//               global_run_id: evalData.global_run_id,
//               created_at: new Date(evalData.created_at as string),
//             },
//           });
//         });
//         const liveEvalPromises = Array.from({ length: 8 }, () => {
//           const evalData = createSampleEval(agentName, false);
//           return store.insert({
//             tableName: TABLE_EVALS,
//             record: {
//               run_id: evalData.run_id,
//               agent_name: evalData.agent_name,
//               input: evalData.input,
//               output: evalData.output,
//               result: evalData.result,
//               metric_name: evalData.metric_name,
//               instructions: evalData.instructions,
//               test_info: evalData.test_info,
//               global_run_id: evalData.global_run_id,
//               created_at: new Date(evalData.created_at as string),
//             },
//           });
//         });
//         await Promise.all([...testEvalPromises, ...liveEvalPromises]);

//         const testResults = await store.getEvals({ agentName, type: 'test', page: 0, perPage: 5 });
//         expect(testResults.evals).toHaveLength(5);
//         expect(testResults.total).toBe(10);

//         const liveResults = await store.getEvals({ agentName, type: 'live', page: 1, perPage: 3 });
//         expect(liveResults.evals).toHaveLength(3);
//         expect(liveResults.total).toBe(8);
//         expect(liveResults.hasMore).toBe(true);
//       });

//       it('should filter by date with pagination for getEvals', async () => {
//         const agentName = 'pagination-agent-date-evals';
//         const now = new Date();
//         const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
//         const dayBeforeYesterday = new Date(now.getTime() - 48 * 60 * 60 * 1000);

//         const createEvalAtDate = (date: Date) => {
//           const evalData = createSampleEval(agentName, false, date); // Pass date to helper
//           return store.insert({
//             tableName: TABLE_EVALS,
//             record: {
//               run_id: evalData.run_id, // Use snake_case from helper
//               agent_name: evalData.agent_name,
//               input: evalData.input,
//               output: evalData.output,
//               result: evalData.result,
//               metric_name: evalData.metric_name,
//               instructions: evalData.instructions,
//               test_info: evalData.test_info,
//               global_run_id: evalData.global_run_id,
//               created_at: evalData.created_at, // Use created_at from helper (already Date or ISO string)
//             },
//           });
//         };

//         await Promise.all([
//           createEvalAtDate(dayBeforeYesterday),
//           createEvalAtDate(dayBeforeYesterday),
//           createEvalAtDate(yesterday),
//           createEvalAtDate(yesterday),
//           createEvalAtDate(yesterday),
//           createEvalAtDate(now),
//           createEvalAtDate(now),
//           createEvalAtDate(now),
//           createEvalAtDate(now),
//         ]);

//         const fromYesterday = await store.getEvals({ agentName, dateRange: { start: yesterday }, page: 0, perPage: 3 });
//         expect(fromYesterday.total).toBe(7); // 3 yesterday + 4 now
//         expect(fromYesterday.evals).toHaveLength(3);
//         // Evals are sorted DESC, so first 3 are from 'now'
//         fromYesterday.evals.forEach(e =>
//           expect(new Date(e.createdAt).getTime()).toBeGreaterThanOrEqual(yesterday.getTime()),
//         );

//         const onlyDayBefore = await store.getEvals({
//           agentName,
//           dateRange: {
//             end: new Date(yesterday.getTime() - 1),
//           },
//           page: 0,
//           perPage: 5,
//         });
//         expect(onlyDayBefore.total).toBe(2);
//         expect(onlyDayBefore.evals).toHaveLength(2);
//       });
//     });

//     describe('getTraces with pagination', () => {
//       it('should return paginated traces with total count', async () => {
//         const tracePromises = Array.from({ length: 18 }, (_, i) =>
//           store.insert({ tableName: TABLE_TRACES, record: createSampleTraceForDB(`test-trace-${i}`, 'pg-test-scope') }),
//         );
//         await Promise.all(tracePromises);

//         const page1 = await store.getTracesPaginated({
//           scope: 'pg-test-scope',
//           page: 0,
//           perPage: 8,
//         });
//         expect(page1.traces).toHaveLength(8);
//         expect(page1.total).toBe(18);
//         expect(page1.page).toBe(0);
//         expect(page1.perPage).toBe(8);
//         expect(page1.hasMore).toBe(true);

//         const page3 = await store.getTracesPaginated({
//           scope: 'pg-test-scope',
//           page: 2,
//           perPage: 8,
//         });
//         expect(page3.traces).toHaveLength(2);
//         expect(page3.total).toBe(18);
//         expect(page3.hasMore).toBe(false);
//       });

//       it('should filter by attributes with pagination for getTraces', async () => {
//         const tracesWithAttr = Array.from({ length: 8 }, (_, i) =>
//           store.insert({
//             tableName: TABLE_TRACES,
//             record: createSampleTraceForDB(`trace-${i}`, 'pg-attr-scope', { environment: 'prod' }),
//           }),
//         );
//         const tracesWithoutAttr = Array.from({ length: 5 }, (_, i) =>
//           store.insert({
//             tableName: TABLE_TRACES,
//             record: createSampleTraceForDB(`trace-other-${i}`, 'pg-attr-scope', { environment: 'dev' }),
//           }),
//         );
//         await Promise.all([...tracesWithAttr, ...tracesWithoutAttr]);

//         const prodTraces = await store.getTracesPaginated({
//           scope: 'pg-attr-scope',
//           attributes: { environment: 'prod' },
//           page: 0,
//           perPage: 5,
//         });
//         expect(prodTraces.traces).toHaveLength(5);
//         expect(prodTraces.total).toBe(8);
//         expect(prodTraces.hasMore).toBe(true);
//       });

//       it('should filter by date with pagination for getTraces', async () => {
//         const scope = 'pg-date-traces';
//         const now = new Date();
//         const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
//         const dayBeforeYesterday = new Date(now.getTime() - 48 * 60 * 60 * 1000);

//         await Promise.all([
//           store.insert({
//             tableName: TABLE_TRACES,
//             record: createSampleTraceForDB('t1', scope, undefined, dayBeforeYesterday),
//           }),
//           store.insert({ tableName: TABLE_TRACES, record: createSampleTraceForDB('t2', scope, undefined, yesterday) }),
//           store.insert({ tableName: TABLE_TRACES, record: createSampleTraceForDB('t3', scope, undefined, yesterday) }),
//           store.insert({ tableName: TABLE_TRACES, record: createSampleTraceForDB('t4', scope, undefined, now) }),
//           store.insert({ tableName: TABLE_TRACES, record: createSampleTraceForDB('t5', scope, undefined, now) }),
//         ]);

//         const fromYesterday = await store.getTracesPaginated({
//           scope,
//           dateRange: {
//             start: yesterday,
//           },
//           page: 0,
//           perPage: 2,
//         });
//         expect(fromYesterday.total).toBe(4); // 2 yesterday + 2 now
//         expect(fromYesterday.traces).toHaveLength(2);
//         fromYesterday.traces.forEach(t =>
//           expect(new Date(t.createdAt).getTime()).toBeGreaterThanOrEqual(yesterday.getTime()),
//         );

//         const onlyNow = await store.getTracesPaginated({
//           scope,
//           dateRange: {
//             start: now,
//             end: now,
//           },
//           page: 0,
//           perPage: 5,
//         });
//         expect(onlyNow.total).toBe(2);
//         expect(onlyNow.traces).toHaveLength(2);
//       });
//     });

//     describe('getMessages with pagination', () => {
//       it('should return paginated messages with total count', async () => {
//         const thread = createSampleThread();
//         await store.saveThread({ thread });
//         // Reset role to 'assistant' before creating messages
//         resetRole();
//         // Create messages sequentially to ensure unique timestamps
//         for (let i = 0; i < 15; i++) {
//           const message = createSampleMessageV1({ threadId: thread.id, content: `Message ${i + 1}` });
//           await store.saveMessages({
//             messages: [message],
//           });
//           await new Promise(r => setTimeout(r, 5));
//         }

//         const page1 = await store.getMessagesPaginated({
//           threadId: thread.id,
//           selectBy: { pagination: { page: 0, perPage: 5 } },
//           format: 'v2',
//         });
//         expect(page1.messages).toHaveLength(5);
//         expect(page1.total).toBe(15);
//         expect(page1.page).toBe(0);
//         expect(page1.perPage).toBe(5);
//         expect(page1.hasMore).toBe(true);

//         const page3 = await store.getMessagesPaginated({
//           threadId: thread.id,
//           selectBy: { pagination: { page: 2, perPage: 5 } },
//           format: 'v2',
//         });
//         expect(page3.messages).toHaveLength(5);
//         expect(page3.total).toBe(15);
//         expect(page3.hasMore).toBe(false);
//       });

//       it('should filter by date with pagination for getMessages', async () => {
//         resetRole();
//         const threadData = createSampleThread();
//         const thread = await store.saveThread({ thread: threadData as StorageThreadType });
//         const now = new Date();
//         const yesterday = new Date(
//           now.getFullYear(),
//           now.getMonth(),
//           now.getDate() - 1,
//           now.getHours(),
//           now.getMinutes(),
//           now.getSeconds(),
//         );
//         const dayBeforeYesterday = new Date(
//           now.getFullYear(),
//           now.getMonth(),
//           now.getDate() - 2,
//           now.getHours(),
//           now.getMinutes(),
//           now.getSeconds(),
//         );

//         // Ensure timestamps are distinct for reliable sorting by creating them with a slight delay for testing clarity
//         const messagesToSave: MastraMessageV1[] = [];
//         messagesToSave.push(createSampleMessageV1({ threadId: thread.id, createdAt: dayBeforeYesterday }));
//         await new Promise(r => setTimeout(r, 5));
//         messagesToSave.push(createSampleMessageV1({ threadId: thread.id, createdAt: dayBeforeYesterday }));
//         await new Promise(r => setTimeout(r, 5));
//         messagesToSave.push(createSampleMessageV1({ threadId: thread.id, createdAt: yesterday }));
//         await new Promise(r => setTimeout(r, 5));
//         messagesToSave.push(createSampleMessageV1({ threadId: thread.id, createdAt: yesterday }));
//         await new Promise(r => setTimeout(r, 5));
//         messagesToSave.push(createSampleMessageV1({ threadId: thread.id, createdAt: now }));
//         await new Promise(r => setTimeout(r, 5));
//         messagesToSave.push(createSampleMessageV1({ threadId: thread.id, createdAt: now }));

//         await store.saveMessages({ messages: messagesToSave, format: 'v1' });
//         // Total 6 messages: 2 now, 2 yesterday, 2 dayBeforeYesterday (oldest to newest)

//         const fromYesterday = await store.getMessagesPaginated({
//           threadId: thread.id,
//           selectBy: { pagination: { page: 0, perPage: 3, dateRange: { start: yesterday } } },
//           format: 'v2',
//         });
//         expect(fromYesterday.total).toBe(4);
//         expect(fromYesterday.messages).toHaveLength(3);
//         const firstMessageTime = new Date((fromYesterday.messages[0] as MastraMessageV1).createdAt).getTime();
//         expect(firstMessageTime).toBeGreaterThanOrEqual(new Date(yesterday.toISOString()).getTime());
//         if (fromYesterday.messages.length > 0) {
//           expect(new Date((fromYesterday.messages[0] as MastraMessageV1).createdAt).toISOString().slice(0, 10)).toEqual(
//             yesterday.toISOString().slice(0, 10),
//           );
//         }
//       });

//       it('should save and retrieve messages', async () => {
//         const thread = createSampleThread();
//         await store.saveThread({ thread });

//         const messages = [
//           createSampleMessageV1({ threadId: thread.id }),
//           createSampleMessageV1({ threadId: thread.id }),
//         ];

//         // Save messages
//         const savedMessages = await store.saveMessages({ messages });
//         // Retrieve messages
//         const retrievedMessages = await store.getMessagesPaginated({ threadId: thread.id, format: 'v1' });

//         const checkMessages = messages.map(m => {
//           const { resourceId, ...rest } = m;
//           return rest;
//         });

//         try {
//           expect(savedMessages).toEqual(messages);
//           expect(retrievedMessages.messages).toHaveLength(2);
//           expect(retrievedMessages.messages).toEqual(expect.arrayContaining(checkMessages));
//         } catch (e) {
//           console.error('Error in should save and retrieve messages:', e);
//           throw e;
//         }
//       });

//       it('should maintain message order', async () => {
//         const thread = createSampleThread();
//         await store.saveThread({ thread });

//         const messageContent = ['First', 'Second', 'Third'];

//         const messages = messageContent.map(content =>
//           createSampleMessageV2({
//             threadId: thread.id,
//             content: { content, parts: [{ type: 'text', text: content }] },
//           }),
//         );

//         await store.saveMessages({ messages, format: 'v2' });

//         const retrievedMessages = await store.getMessagesPaginated({ threadId: thread.id, format: 'v2' });
//         expect(retrievedMessages.messages).toHaveLength(3);

//         // Verify order is maintained
//         retrievedMessages.messages.forEach((msg, idx) => {
//           if (typeof msg.content === 'object' && msg.content && 'parts' in msg.content) {
//             expect((msg.content.parts[0] as any).text).toEqual(messageContent[idx]);
//           } else {
//             throw new Error('Message content is not in expected format');
//           }
//         });
//       });

//       it('should rollback on error during message save', async () => {
//         const thread = createSampleThread();
//         await store.saveThread({ thread });

//         const messages = [
//           createSampleMessageV1({ threadId: thread.id }),
//           { ...createSampleMessageV1({ threadId: thread.id }), id: null } as any, // This will cause an error
//         ];

//         await expect(store.saveMessages({ messages })).rejects.toThrow();

//         // Verify no messages were saved
//         const savedMessages = await store.getMessagesPaginated({ threadId: thread.id, format: 'v2' });
//         expect(savedMessages.messages).toHaveLength(0);
//       });

//       it('should retrieve messages w/ next/prev messages by message id + resource id', async () => {
//         const thread = createSampleThread({ id: 'thread-one' });
//         await store.saveThread({ thread });

//         const thread2 = createSampleThread({ id: 'thread-two' });
//         await store.saveThread({ thread: thread2 });

//         const thread3 = createSampleThread({ id: 'thread-three' });
//         await store.saveThread({ thread: thread3 });

//         const messages: MastraMessageV2[] = [
//           createSampleMessageV2({
//             threadId: 'thread-one',
//             content: { content: 'First' },
//             resourceId: 'cross-thread-resource',
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-one',
//             content: { content: 'Second' },
//             resourceId: 'cross-thread-resource',
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-one',
//             content: { content: 'Third' },
//             resourceId: 'cross-thread-resource',
//           }),

//           createSampleMessageV2({
//             threadId: 'thread-two',
//             content: { content: 'Fourth' },
//             resourceId: 'cross-thread-resource',
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-two',
//             content: { content: 'Fifth' },
//             resourceId: 'cross-thread-resource',
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-two',
//             content: { content: 'Sixth' },
//             resourceId: 'cross-thread-resource',
//           }),

//           createSampleMessageV2({
//             threadId: 'thread-three',
//             content: { content: 'Seventh' },
//             resourceId: 'other-resource',
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-three',
//             content: { content: 'Eighth' },
//             resourceId: 'other-resource',
//           }),
//         ];

//         await store.saveMessages({ messages: messages, format: 'v2' });

//         const retrievedMessages = await store.getMessagesPaginated({ threadId: 'thread-one', format: 'v2' });
//         expect(retrievedMessages.messages).toHaveLength(3);
//         expect(retrievedMessages.messages.map((m: any) => m.content.parts[0].text)).toEqual([
//           'First',
//           'Second',
//           'Third',
//         ]);

//         const retrievedMessages2 = await store.getMessagesPaginated({ threadId: 'thread-two', format: 'v2' });
//         expect(retrievedMessages2.messages).toHaveLength(3);
//         expect(retrievedMessages2.messages.map((m: any) => m.content.parts[0].text)).toEqual([
//           'Fourth',
//           'Fifth',
//           'Sixth',
//         ]);

//         const retrievedMessages3 = await store.getMessagesPaginated({ threadId: 'thread-three', format: 'v2' });
//         expect(retrievedMessages3.messages).toHaveLength(2);
//         expect(retrievedMessages3.messages.map((m: any) => m.content.parts[0].text)).toEqual(['Seventh', 'Eighth']);

//         const { messages: crossThreadMessages } = await store.getMessagesPaginated({
//           threadId: 'thread-doesnt-exist',
//           format: 'v2',
//           selectBy: {
//             last: 0,
//             include: [
//               {
//                 id: messages[1].id,
//                 threadId: 'thread-one',
//                 withNextMessages: 2,
//                 withPreviousMessages: 2,
//               },
//               {
//                 id: messages[4].id,
//                 threadId: 'thread-two',
//                 withPreviousMessages: 2,
//                 withNextMessages: 2,
//               },
//             ],
//           },
//         });
//         expect(crossThreadMessages).toHaveLength(6);
//         expect(crossThreadMessages.filter(m => m.threadId === `thread-one`)).toHaveLength(3);
//         expect(crossThreadMessages.filter(m => m.threadId === `thread-two`)).toHaveLength(3);
//       });

//       it('should return messages using both last and include (cross-thread, deduped)', async () => {
//         const thread = createSampleThread({ id: 'thread-one' });
//         await store.saveThread({ thread });

//         const thread2 = createSampleThread({ id: 'thread-two' });
//         await store.saveThread({ thread: thread2 });

//         const now = new Date();

//         // Setup: create messages in two threads
//         const messages = [
//           createSampleMessageV2({
//             threadId: 'thread-one',
//             content: { content: 'A' },
//             createdAt: new Date(now.getTime()),
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-one',
//             content: { content: 'B' },
//             createdAt: new Date(now.getTime() + 1000),
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-one',
//             content: { content: 'C' },
//             createdAt: new Date(now.getTime() + 2000),
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-two',
//             content: { content: 'D' },
//             createdAt: new Date(now.getTime() + 3000),
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-two',
//             content: { content: 'E' },
//             createdAt: new Date(now.getTime() + 4000),
//           }),
//           createSampleMessageV2({
//             threadId: 'thread-two',
//             content: { content: 'F' },
//             createdAt: new Date(now.getTime() + 5000),
//           }),
//         ];
//         await store.saveMessages({ messages, format: 'v2' });

//         // Use last: 2 and include a message from another thread with context
//         const { messages: result } = await store.getMessagesPaginated({
//           threadId: 'thread-one',
//           format: 'v2',
//           selectBy: {
//             last: 2,
//             include: [
//               {
//                 id: messages[4].id, // 'E' from thread-bar
//                 threadId: 'thread-two',
//                 withPreviousMessages: 1,
//                 withNextMessages: 1,
//               },
//             ],
//           },
//         });

//         // Should include last 2 from thread-one and 3 from thread-two (D, E, F)
//         expect(result.map(m => (m.content as { content: string }).content).sort()).toEqual(['B', 'C', 'D', 'E', 'F']);
//         // Should include 2 from thread-one
//         expect(result.filter(m => m.threadId === 'thread-one').map((m: any) => m.content.content)).toEqual(['B', 'C']);
//         // Should include 3 from thread-two
//         expect(result.filter(m => m.threadId === 'thread-two').map((m: any) => m.content.content)).toEqual([
//           'D',
//           'E',
//           'F',
//         ]);
//       });
//     });

//     describe('getThreadsByResourceId with pagination', () => {
//       it('should return paginated threads with total count', async () => {
//         const resourceId = `pg-paginated-resource-${randomUUID()}`;
//         const threadPromises = Array.from({ length: 17 }, () =>
//           store.saveThread({ thread: { ...createSampleThread(), resourceId } }),
//         );
//         await Promise.all(threadPromises);

//         const page1 = await store.getThreadsByResourceIdPaginated({ resourceId, page: 0, perPage: 7 });
//         expect(page1.threads).toHaveLength(7);
//         expect(page1.total).toBe(17);
//         expect(page1.page).toBe(0);
//         expect(page1.perPage).toBe(7);
//         expect(page1.hasMore).toBe(true);

//         const page3 = await store.getThreadsByResourceIdPaginated({ resourceId, page: 2, perPage: 7 });
//         expect(page3.threads).toHaveLength(3); // 17 total, 7 per page, 3rd page has 17 - 2*7 = 3
//         expect(page3.total).toBe(17);
//         expect(page3.hasMore).toBe(false);
//       });

//       it('should return paginated results when no pagination params for getThreadsByResourceId', async () => {
//         const resourceId = `pg-non-paginated-resource-${randomUUID()}`;
//         await store.saveThread({ thread: { ...createSampleThread(), resourceId } });

//         const results = await store.getThreadsByResourceIdPaginated({ resourceId });
//         expect(Array.isArray(results.threads)).toBe(true);
//         expect(results.threads.length).toBe(1);
//         expect(results.total).toBe(1);
//         expect(results.page).toBe(0);
//         expect(results.perPage).toBe(100);
//         expect(results.hasMore).toBe(false);
//       });
//     });
//   });

//   describe('MssqlStorage Table Name Quoting', () => {
//     const camelCaseTable = 'TestCamelCaseTable';
//     const snakeCaseTable = 'test_snake_case_table';
//     const BASE_SCHEMA = {
//       id: { type: 'integer', primaryKey: true, nullable: false },
//       name: { type: 'text', nullable: true },
//     } as Record<string, StorageColumn>;

//     beforeEach(async () => {
//       // Only clear tables if store is initialized
//       try {
//         // Clear tables before each test
//         await store.clearTable({ tableName: camelCaseTable as TABLE_NAMES });
//         await store.clearTable({ tableName: snakeCaseTable as TABLE_NAMES });
//       } catch (error) {
//         // Ignore errors during table clearing
//         console.warn('Error clearing tables:', error);
//       }
//     });

//     afterEach(async () => {
//       // Only clear tables if store is initialized
//       try {
//         // Clear tables before each test
//         await store.clearTable({ tableName: camelCaseTable as TABLE_NAMES });
//         await store.clearTable({ tableName: snakeCaseTable as TABLE_NAMES });
//       } catch (error) {
//         // Ignore errors during table clearing
//         console.warn('Error clearing tables:', error);
//       }
//     });

//     it('should create and upsert to a camelCase table without quoting errors', async () => {
//       await expect(
//         store.createTable({
//           tableName: camelCaseTable as TABLE_NAMES,
//           schema: BASE_SCHEMA,
//         }),
//       ).resolves.not.toThrow();

//       await store.insert({
//         tableName: camelCaseTable as TABLE_NAMES,
//         record: { id: '1', name: 'Alice' },
//       });

//       const row: any = await store.load({
//         tableName: camelCaseTable as TABLE_NAMES,
//         keys: { id: '1' },
//       });
//       expect(row?.name).toBe('Alice');
//     });

//     it('should create and upsert to a snake_case table without quoting errors', async () => {
//       await expect(
//         store.createTable({
//           tableName: snakeCaseTable as TABLE_NAMES,
//           schema: BASE_SCHEMA,
//         }),
//       ).resolves.not.toThrow();

//       await store.insert({
//         tableName: snakeCaseTable as TABLE_NAMES,
//         record: { id: '2', name: 'Bob' },
//       });

//       const row: any = await store.load({
//         tableName: snakeCaseTable as TABLE_NAMES,
//         keys: { id: '2' },
//       });
//       expect(row?.name).toBe('Bob');
//     });
//   });

//   describe('Permission Handling (MSSQL)', () => {
//     const schemaRestrictedUser = 'mastra_schema_restricted_storage';
//     const restrictedPassword = 'Test123!@#'; // MSSQL requires a strong password
//     const testSchema = 'testSchema';
//     const adminConfig = {
//       user: TEST_CONFIG.user,
//       password: TEST_CONFIG.password,
//       server: TEST_CONFIG.server,
//       database: TEST_CONFIG.database,
//       port: TEST_CONFIG.port,
//       options: { encrypt: true, trustServerCertificate: true },
//     };

//     let adminPool: sql.ConnectionPool;

//     beforeAll(async () => {
//       adminPool = await sql.connect(adminConfig);

//       // Drop schema and user if they exist
//       await adminPool.request().batch(`
//         IF EXISTS (SELECT * FROM sys.schemas WHERE name = '${testSchema}')
//           DROP SCHEMA [${testSchema}];
//         IF EXISTS (SELECT * FROM sys.database_principals WHERE name = '${schemaRestrictedUser}')
//           DROP USER [${schemaRestrictedUser}];
//         IF EXISTS (SELECT * FROM sys.sql_logins WHERE name = '${schemaRestrictedUser}')
//           DROP LOGIN [${schemaRestrictedUser}];
//       `);

//       // Create restricted login and user
//       await adminPool.request().batch(`
//         CREATE LOGIN [${schemaRestrictedUser}] WITH PASSWORD = '${restrictedPassword}';
//         CREATE USER [${schemaRestrictedUser}] FOR LOGIN [${schemaRestrictedUser}];
//         -- Only grant CONNECT, do not grant CREATE SCHEMA
//         GRANT CONNECT TO [${schemaRestrictedUser}];
//       `);
//     });

//     afterAll(async () => {
//       // Drop schema and user
//       await adminPool.request().batch(`
//         IF EXISTS (SELECT * FROM sys.schemas WHERE name = '${testSchema}')
//           DROP SCHEMA [${testSchema}];
//         IF EXISTS (SELECT * FROM sys.database_principals WHERE name = '${schemaRestrictedUser}')
//           DROP USER [${schemaRestrictedUser}];
//         IF EXISTS (SELECT * FROM sys.sql_logins WHERE name = '${schemaRestrictedUser}')
//           DROP LOGIN [${schemaRestrictedUser}];
//       `);
//       await adminPool.close();
//     });

//     describe('Schema Creation', () => {
//       it('should fail when user lacks CREATE SCHEMA privilege', async () => {
//         const restrictedConfig = {
//           ...adminConfig,
//           user: schemaRestrictedUser,
//           password: restrictedPassword,
//         };
//         const store = new MSSQLStore({
//           ...restrictedConfig,
//           schemaName: testSchema,
//         });

//         try {
//           await expect(store.init()).rejects.toThrow(
//             `Unable to create schema "testSchema". This requires CREATE privilege on the database. Either create the schema manually or grant CREATE privilege to the user.`,
//           );

//           // Verify schema was not created
//           const result = await adminPool.request().query(`SELECT * FROM sys.schemas WHERE name = '${testSchema}'`);
//           expect(result.recordset.length).toBe(0);
//         } finally {
//           await store.close();
//         }
//       });

//       it('should fail with schema creation error when saving thread', async () => {
//         const restrictedConfig = {
//           ...adminConfig,
//           user: schemaRestrictedUser,
//           password: restrictedPassword,
//         };
//         const store = new MSSQLStore({
//           ...restrictedConfig,
//           schemaName: testSchema,
//         });

//         try {
//           await expect(async () => {
//             await store.init();
//             const thread = createSampleThread();
//             await store.saveThread({ thread });
//           }).rejects.toThrow(
//             `Unable to create schema "testSchema". This requires CREATE privilege on the database. Either create the schema manually or grant CREATE privilege to the user.`,
//           );

//           // Verify schema was not created
//           const result = await adminPool.request().query(`SELECT * FROM sys.schemas WHERE name = '${testSchema}'`);
//           expect(result.recordset.length).toBe(0);
//         } finally {
//           await store.close();
//         }
//       });
//     });
//   });

//   afterAll(async () => {
//     try {
//       await store.close();
//     } catch (error) {
//       console.warn('Error closing store:', error);
//     }
//   });
// });
