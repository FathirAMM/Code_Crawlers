import React, { useState } from "react";
import { Form, Button, FormControl, ProgressBar } from "react-bootstrap";
import FileImage from 'bootstrap-icons/icons/file-image.svg';
import FileEarmark from 'bootstrap-icons/icons/file-earmark.svg';
import FilePdf from 'bootstrap-icons/icons/file-pdf.svg';
import './chat/fileupload.css';

async function injectToVectorDB(file, database, onProgress, urls) {
  const url = "http://127.0.0.1:5000/api/inject_to_vector_db";
  const formData = new FormData();
  //formData.append('file', file);
  //formData.append('database', database);
  //formData.append('url',urls)
  //console.log(urls);
  //console.log(formData['url']);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        file: file,
        urls: urls,
        database: database,

      })
      // formData,
      //onUploadProgress: (progressEvent) => {
      //  const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
      //  onProgress(progress);
      //}
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const responseData = await response.json();
    return responseData;
  } catch (error) {
    console.error("Error in injectToVectorDB:", error);
    return null;
  }
}

function InjectToVectorDB() {
  const [dbStatus, setDbStatus] = useState("");
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    setUploadedFile(file);
  };

  const handleDbSubmit = async () => {
    const database = document.querySelector('input[name="dbChoice"]:checked').value;
    const file = uploadedFile;
    const urls = document.getElementById('url').value;
    console.log(file);
    const response = await injectToVectorDB(file, database, (progress) => {
      setUploadProgress(progress);
    }, urls);

    if (response) {
      setDbStatus(response.message);
    } else {
      setDbStatus("Error injecting data to vector DB.");
    }
  };
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setUploadedFile(selectedFile);
    // setFile(selectedFile);
    // You can perform additional actions with the selected file, such as uploading it to a server
  };
  return (
    <div style={{ height: "1000px" }}>
      <div className="wrapper">
        <div className="container">
          <h3>Upload a file</h3>
          <div className="upload-container"
            onDragOver={handleDragOver}
            onDrop={handleDrop}>
            <div className="border-container">
              {uploadedFile ? (
                <p>Uploaded file: {uploadedFile.name}</p>
              ) : (
                <div>
                  <div className="icons fa-4x">
                    <svg xmlns="http://www.w3.org/2000/svg" iwidth="64" height="64" fill="black" class="bi bi-file-image" viewBox="0 0 16 16">
                      <path d="M8.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0"></path>
                      <path d="M12 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2M3 2a1 1 0 0 1 1-1h8a1 1 0 0 1 1 1v8l-2.083-2.083a.5.5 0 0 0-.76.063L8 11 5.835 9.7a.5.5 0 0 0-.611.076L3 12z"></path>
                    </svg><svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="black" class="bi bi-file-earmark" viewBox="0 0 16 16">
                      <path d="M14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h5.5zm-3 0A1.5 1.5 0 0 1 9.5 3V1H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V4.5z"></path>
                    </svg><svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="black" class="bi bi-file-pdf" viewBox="0 0 16 16">
                      <path d="M4 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zm0 1h8a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1"></path>
                      <path d="M4.603 12.087a.8.8 0 0 1-.438-.42c-.195-.388-.13-.776.08-1.102.198-.307.526-.568.897-.787a7.7 7.7 0 0 1 1.482-.645 20 20 0 0 0 1.062-2.227 7.3 7.3 0 0 1-.43-1.295c-.086-.4-.119-.796-.046-1.136.075-.354.274-.672.65-.823.192-.077.4-.12.602-.077a.7.7 0 0 1 .477.365c.088.164.12.356.127.538.007.187-.012.395-.047.614-.084.51-.27 1.134-.52 1.794a11 11 0 0 0 .98 1.686 5.8 5.8 0 0 1 1.334.05c.364.065.734.195.96.465.12.144.193.32.2.518.007.192-.047.382-.138.563a1.04 1.04 0 0 1-.354.416.86.86 0 0 1-.51.138c-.331-.014-.654-.196-.933-.417a5.7 5.7 0 0 1-.911-.95 11.6 11.6 0 0 0-1.997.406 11.3 11.3 0 0 1-1.021 1.51c-.29.35-.608.655-.926.787a.8.8 0 0 1-.58.029m1.379-1.901q-.25.115-.459.238c-.328.194-.541.383-.647.547-.094.145-.096.25-.04.361q.016.032.026.044l.035-.012c.137-.056.355-.235.635-.572a8 8 0 0 0 .45-.606m1.64-1.33a13 13 0 0 1 1.01-.193 12 12 0 0 1-.51-.858 21 21 0 0 1-.5 1.05zm2.446.45q.226.244.435.41c.24.19.407.253.498.256a.1.1 0 0 0 .07-.015.3.3 0 0 0 .094-.125.44.44 0 0 0 .059-.2.1.1 0 0 0-.026-.063c-.052-.062-.2-.152-.518-.209a4 4 0 0 0-.612-.053zM8.078 5.8a7 7 0 0 0 .2-.828q.046-.282.038-.465a.6.6 0 0 0-.032-.198.5.5 0 0 0-.145.04c-.087.035-.158.106-.196.283-.04.192-.03.469.046.822q.036.167.09.346z"></path>
                    </svg>
                    <p className="bi">csv/pdf/png/jpeg</p>
                  </div>
                  <p className="bi">Drag and drop files here, or&ensp;
                    <label htmlFor="file-upload" id="file-browser">browse</label> your computer.
                  </p>
                  <input id="file-upload" type="file" style={{ display: 'none' }} onChange={handleFileChange} />

                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      <br />
      <div id="containt1" style={{
        width: "82.55rem",
        /* margin-left: 12.1rem; */
        /* margin-top: -1rem; */
        padding: "34px",
        margin: "auto",
        marginTop: "-2%",
        background: "rgb(239, 239, 239)",
      }
      }>
        <Form.Label>Choose Database</Form.Label>
        <br />
        <Form.Group style={{ display: "flex", gap: "2.1rem" }}>
          <Form.Check type="radio" label="Faiss" name="dbChoice" value="faiss" />
          <Form.Check type="radio" label="Pinecone" name="dbChoice" value="pinecone" />
        </Form.Group>
        <br />
        <Form.Label>Import Url</Form.Label>
        <br />
        <Form.Control id="url" type="text" />
        <br />
        <Button style={{ width: '100%' }} onClick={handleDbSubmit}>Inject Data</Button>
        <br />
        <br />
        {/* <ProgressBar animated now={uploadProgress} label={`${uploadProgress}%`} /> */}
        <FormControl placeholder="Status" value={dbStatus} readOnly />
      </div>
    </div>
  );
}

export default InjectToVectorDB;
