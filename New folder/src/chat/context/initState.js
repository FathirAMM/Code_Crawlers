export const initState = {
  conversation: [],
  current: 0,
  chat: [
    
    {
      title: "ex",
      ct: "2032-12-23",
      id: 2381923,
      messages: [],
    },
  ],
  currentChat: 0,
  options: {
    account: {
      name: "CHAT——AI",
      avatar: "",
    },
    general: {
      language: "English",
      theme: "light",
      command: "COMMAND_ENTER",
      size: "normal",
    },
    openai: {
      baseUrl: "",
      organizationId: "",
      temperature: 1,
      model: "gpt-3.5-turbo",
      database: "faiss",
      apiKey: "apikey",
      max_tokens: 2048,
      n: 1,
      stream: true,
    },
  },
  is: {
    typeing: false,
    config: false,
    fullScreen: true,
    sidebar: true,
    inputing: false,
    thinking: false,
    apps: true,
  },
  typeingMessage: {},
  version: "0.1.0",
  cotent: "",
};
