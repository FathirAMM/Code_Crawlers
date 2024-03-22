import React, { useState } from "react";
import { Form, Button, FormControl,ProgressBar } from "react-bootstrap";
import FileImage from 'bootstrap-icons/icons/file-image.svg';
import FileEarmark from 'bootstrap-icons/icons/file-earmark.svg';
import FilePdf from 'bootstrap-icons/icons/file-pdf.svg';
import './chat/fileupload.css';
function showPopup() {
  // Get the popup element
  var popup = document.getElementById("popup");

  // Display the popup
  popup.style.display = "block";

  // Center the popup vertically
  var windowHeight = window.innerHeight;
  var popupHeight = popup.offsetHeight;
  popup.style.top = ((windowHeight - popupHeight) / 2) + "px";
}


async function Finetuned(file, modelname,apikey) {
  const url = "http://127.0.0.1:5000//api/finetune";
  const formData = new FormData();
  // file = request.files["file"]
  //   suffix = request.form["suffixName"]
  //   api_key = request.form["apiKey"]
  formData.append('file', file);
  formData.append('suffixName', modelname);
  formData.append('apiKey', apikey);

  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const responseData = await response.json();
    // console.log(response);
    return responseData;
  } catch (error) {
    console.error("Error in injectToVectorDB:", error);
    return null;
  }
}

function Finetunedd() {
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
    const file = document.getElementById('fileInput1').files[0];
    // console.log(document.getElementById('fileInput1').files[0]);
    const suffixName = document.getElementById('modelname').value;
    // console.log(document.getElementById('modelname').value);
    const apiKey = document.getElementById('apikey').value;
    // console.log(document.getElementById('apikey').value);
    document.getElementById('evaluvate').style.display = 'flex';
    const response = await Finetuned(file, suffixName,apiKey,(progress) => {
      setUploadProgress(progress)});
      
    if (response) {
      setDbStatus(response.message);
      
    } else {
      alert("model finetuned"+"modelname"+suffixName);
      setDbStatus("Error finetune the model.");
    }
    
  };
  // const showPopup = async () => {
  //   const popup = document.createElement('div');
  //   popup.className = 'popup';
  //   popup.innerText = 'Evaluation in progress...';
  
  //   document.body.appendChild(popup);
  
  //   setTimeout(() => {
  //     popup.remove();
  //   }, 3000); // Remove popup after 3 seconds (adjust as needed)
  // };
  
  // // Add event listener to the 'evaluate' button
  // document.getElementById('evaluvate').addEventListener('click', showPopup);
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setUploadedFile(selectedFile);
    // You can perform additional actions with the selected file, such as uploading it to a server
  };
  return (
    
    <div style={{height : "1000px"}}>
      <div id="popup" className="popup" style={{width:"50%",display:'none',position:"fixed"}}>
        <img />
      </div>
      <div className="wrapper">
        <div className="container">
          <h4>Upload a file</h4>
          <div className="upload-container"
            onDragOver={handleDragOver}
            onDrop={handleDrop}>
            <div className="border-container">
              {uploadedFile ? (
                <p>Uploaded file: {uploadedFile.name}</p>
              ) : (
                <div>
                  <div className="icons fa-4x">
                  <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="black" class="bi bi-file-earmark" viewBox="0 0 16 16">
                      <path d="M14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h5.5zm-3 0A1.5 1.5 0 0 1 9.5 3V1H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V4.5z"></path></svg>
                   <p className="bi">csv</p>
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
    
      <div id="containt" style={{
        width: "82.55rem",
        /* margin-left: 12.1rem; */
        /* margin-top: -1rem; */
        padding: "34px",
        margin: "auto",
        marginTop: "-2%",
        background: "rgb(239, 239, 239)",
    }
      }>
      {/* <Form.Group>
        <Form.Label>Upload PDF</Form.Label>
        <Form.Control id="fileInput1" type="file" />
      </Form.Group> */}
      <Form.Group>
        <Form.Label>Enter the Model Name</Form.Label>
        <Form.Control type="text" name="modelname" id="modelname"/>
        <br />
        <Form.Label >Enter api Key</Form.Label>
        <Form.Control type="text" name="apikey" id="apikey"/>
        <br />
</Form.Group>
<Button style={{ width: '100%' }} onClick={handleDbSubmit}>Inject Data</Button>
        <br /><br />
        {/* <ProgressBar animated now={uploadProgress} label={`${uploadProgress}%`} />
         */}
      <FormControl placeholder="Status" value={dbStatus} readOnly />
      <br />
      <Button id="evaluvate" style={{ width: '100%' }} onClick={showPopup}>Evaluate</Button>
    </div>
    </div>
  );
}

export default Finetunedd;
