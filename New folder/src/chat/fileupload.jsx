import React, { useState } from 'react';
import FileImage from 'bootstrap-icons/icons/file-image.svg';
import FileEarmark from 'bootstrap-icons/icons/file-earmark.svg';
import FilePdf from 'bootstrap-icons/icons/file-pdf.svg';
import './fileupload.css';

export function FileUploader() {
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    setUploadedFile(file);
  };

  return (
    <div className="wrapper">
      <div className="container">
        <h3>Upload a file</h3>
        <div className="upload-container"
          onDragOver={handleDragOver}
          onDrop={handleDrop}>
          <div className="border-container">
          {uploadedFile ? (
              <p>Uploaded file: {uploadedFile.name}</p>
            ) :(
              <div>
            <div className="icons fa-4x">
              <img width={"15%"} src={FileImage} alt="File Image" />
              <img width={"15%"} src={FileEarmark} alt="File Earmark" />
              <img width={"15%"} src={FilePdf} alt="File PDF" />
            </div>
             
              <p>Drag and drop files here, or <a href="#" id="file-browser">browse</a> your computer.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default FileUploader;
