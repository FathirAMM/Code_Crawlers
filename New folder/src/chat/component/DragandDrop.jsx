import React, { useState } from 'react';
import './drag-drop.css';

function FileUploader() {
  const [files, setFiles] = useState([]);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles([...files, ...droppedFiles]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      style={{ width: '100%', height: '200px', border: '2px dashed #ccc' }}
    >
      <h3>Drag & Drop Files Here</h3>
      {files.map((file, index) => (
        <div key={index}>{file.name}</div>
      ))}
    </div>
  );
}

export default FileUploader;
