import { useState } from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [fileName, setFileName] = useState("");
  const [answer, setAnswer] = useState("");

  const uploadPdf = async () => {
    if (!file) return alert("Select a PDF");

    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post(`${API_BASE}/upload_pdf`, formData);
    setFileName(res.data.file);
    alert("PDF uploaded successfully");
  };

  const askQuestion = async () => {
  if (!question || !fileName) return alert("Upload PDF first");

  setAnswer("");

  const response = await fetch(
    `${API_BASE}/ask_stream?question=${question}&file_name=${fileName}`,
    { method: "POST" }
  );

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    setAnswer(prev => prev + decoder.decode(value));
  }
};

  
  

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h2>RAG PDF Question Answering</h2>

      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <br /><br />
      <button onClick={uploadPdf}>Upload PDF</button>

      <hr />

      <input
        type="text"
        placeholder="Ask a question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "60%" }}
      />
      <br /><br />
      <button onClick={askQuestion}>Ask</button>

      <hr />

      <h3>Answer:</h3>
      <p>{answer}</p>
    </div>
  );
}

export default App;
