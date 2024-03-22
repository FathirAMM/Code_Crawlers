import { useState, useEffect } from "react";
import { useGlobal } from "../context";

export function useMesssage() {
  const { currentChat, chat, is } = useGlobal();
  const [message, setMessage] = useState({ messages: [] });
  const [messaged, setMessaged] = useState({ messages: [] });
  useEffect(() => {
    if (chat.length) {
      setMessage(chat[currentChat]);
    }
  }, [chat, is.thinking, currentChat]);
  return { message };
}
export function useMessaged() {
  const { currentChat, chat, is } = useGlobal();
  const [messaged, setMessaged] = useState({ messages: [] });
  useEffect(() => {
    if (chat.length) {
      setMessaged(chat[currentChat]);
    }
  }, [chat, currentChat]);
  return { messaged };
}