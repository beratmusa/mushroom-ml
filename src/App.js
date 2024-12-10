import React, { useState } from "react";
import InputForm from "./components/InputForm";
import Result from "./components/Result";

function App() {
  const [result, setResult] = useState(null);

  const handlePrediction = (result) => {
    setResult(result);
  };

  return (
    <div>
      <h1>Mushroom Classification</h1>
      <InputForm onPrediction={handlePrediction} />
      <Result {...result} />
    </div>
  );
}

export default App;
