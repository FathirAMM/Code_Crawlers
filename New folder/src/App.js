import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Tab, Tabs } from "react-bootstrap";
//import ChatWithPDFs from "./ChatWithPdf";
import InjectToVectorDB from "./InjectToVectorDB";
import RAGApplication from "./RAGApplication";
import Chat from "./chat/Chat";

function App() {
  return (
    <div>
      <h1>MULTIMODAL LLM APPLICATION</h1>
      <Tabs defaultActiveKey="chat">
        <Tab eventKey="chat" title="Chat with PDFs">
          <ChatWithPDFs />
        </Tab>
        <Tab eventKey="db" title="Inject to Vector DB">
          <InjectToVectorDB />
        </Tab>
        <Tab eventKey="rag" title="RAG Application">
          <RAGApplication />
        </Tab>
      </Tabs>
    </div>
  );
}

export default App;
