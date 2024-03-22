export const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
export const shortcutKey = isMac ? "Command+Enter" : "Ctrl+Enter";
export const keyborad = {
  Command: "Window",
  Option: "Alt",
  Control: "Ctrl",
  Shift: "Shift",
};

export const keyboradArray = isMac
  ? Object.keys(keyborad)
  : Object.values(keyborad);
export const themeOptions = [
  {
    label: "Auto",
    value: "auto",
  },
  {
    label: "Light",
    value: "light",
  },
  {
    label: "Dark",
    value: "dark",
  },
];

export const sendCommandOptions = [
  {
    label: "Enter",
    value: "ENTER",
  },
  {
    label: shortcutKey,
    value: "COMMAND_ENTER",
  },
  {
    label: shortcutKey,
    value: "ALT_ENTER",
  },
];

export const modelOptions = [
  {
    label: "gpt-3.5-turbo",
    value: "gpt-3.5-turbo",
  },
  {
    label: "gpt-4",
    value: "gpt-4",
  },
  {
    label: "claude-2",
    value: "claude-2",
  },
  {
    label: "ft:gpt-3.5-turbo",
    value: "ft:gpt-3.5-turbo-0613:veracityai:codecralers-test:8v28lc2C",
  },
  {
    label: "mistral",
    value: "mistral",
  },
  {
    label: "llama2",
    value: "llama2-7b",
  },
  {
    label: "gemma",
    value: "gemma-7b",
  },
   
];
export const dbOptions = [
  {
    label: "pinecone",
    value: "pinecone",
  },
  {
    label: "faiss",
    value: "faiss",
  },
];

export const languageOptions = [
  {
    label: "English",
    value: "en",
  },
  
];

export const sizeOptions = [
  {
    label: "Small",
    value: "small",
  },
  {
    label: "Default",
    value: "default",
  },
  {
    label: "Middle",
    value: "middle",
  },
  {
    label: "Large",
    value: "large",
  },
];
