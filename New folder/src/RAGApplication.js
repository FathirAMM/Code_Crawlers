import React, { useState } from "react";
import { Form, Button, FormControl, Image, Tab, Tabs } from "react-bootstrap";
import Chat from "./chat/Chat";
import { ChatSideBar } from "./chat/ChatSideBar";
function RAGApplication() {
  const [ragAnswer, setRagAnswer] = useState("");
  const [ragImage, setRagImage] = useState("");

  const handleRagSubmit = () => {
    // Implement your logic for handling RAG submission
  };

  return (
    <Tabs defaultActiveKey="rag" id="rag-tab">
      <Tab eventKey="rag" title="RAG Application">
        <div>
          <Form.Group>
            <Form.Label>Enter your question</Form.Label>
            <Form.Control type="text" placeholder="Enter your question" />
          </Form.Group>
          <Form.Group>
            <Form.Label>Select database</Form.Label>
            <Form.Check type="radio" label="Pinecone" name="Pinecone" />
            <Form.Check type="radio" label="Faiss" name="Faiss" />
          </Form.Group>
          <Form.Group>
            <Form.Label>Select LLM mode</Form.Label>
            <Form.Check type="radio" label="gpt-4" name="gpt-4" />
            <Form.Check
              type="radio"
              label="gpt-3.5-turbo"
              name="gpt-3.5-turbo"
            />
            <Form.Check type="radio" label="claude-2" name="claude-2" />
          </Form.Group>
          <Button onClick={handleRagSubmit}>Submit</Button>
          <FormControl placeholder="Answer" value={ragAnswer} readOnly />
          <Image src={ragImage} />
        </div>
      </Tab>
    </Tabs>
  );
}

export default RAGApplication;
