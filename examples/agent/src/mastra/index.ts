import { Mastra } from '@mastra/core';
import { PinoLogger } from '@mastra/loggers';
import { LibSQLStore } from '@mastra/libsql';

import { chefAgent, chefAgentResponses, dynamicAgent } from './agents/index';
import { myMcpServer, myMcpServerTwo } from './mcp/server';
import { myWorkflow } from './workflows';

console.log('Breakpoint on line 9');
console.log('Breakpoint on line 10');
console.log('Breakpoint on line 11');
console.log('Breakpoint on line 12');
console.log('Breakpoint on line 13');
console.log('Breakpoint on line 14');
console.log('Breakpoint on line 15');

const storage = new LibSQLStore({
  url: 'file:./mastra.db',
});

export const mastra = new Mastra({
  agents: { chefAgent, chefAgentResponses, dynamicAgent },
  logger: new PinoLogger({ name: 'Chef', level: 'debug' }),
  storage,
  mcpServers: {
    myMcpServer,
    myMcpServerTwo,
  },
  workflows: { myWorkflow },
  bundler: {
    sourcemap: true,
  },
  serverMiddleware: [
    {
      handler: (c, next) => {
        console.log('Middleware called');
        return next();
      },
    },
  ],
  // telemetry: {
  //   enabled: false,
  // }
});
