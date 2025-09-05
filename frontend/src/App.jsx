import { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [inputURL, setInputURL] = useState("");
  const [outputURL, setOutputURL] = useState("");

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setOutputURL("");
    if (f) setInputURL(URL.createObjectURL(f));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("http://localhost:8000/reconstruct", {
      method: "POST",
      body: fd,
    });
    const blob = await res.blob();
    setOutputURL(URL.createObjectURL(blob));
  };

  return (
    <div style={{ maxWidth: 800, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>Autoencoder Demo</h1>
      <form onSubmit={onSubmit}>
        <input type="file" accept="image/*" onChange={onFileChange} />
        <button type="submit" style={{ marginLeft: 8 }} disabled={!file}>
          Reconstruct
        </button>
      </form>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 24 }}>
        <div>
          <h3>Input</h3>
          {inputURL && <img src={inputURL} alt="input" style={{ width: "100%", objectFit: "contain" }} />}
        </div>
        <div>
          <h3>Reconstruction</h3>
          {outputURL && <img src={outputURL} alt="recon" style={{ width: "100%", objectFit: "contain" }} />}
        </div>
      </div>
    </div>
  );
}
