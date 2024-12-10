import React, { useState } from "react";
import { predictMushroom } from "../api/predict";

function InputForm({ onPrediction }) {
  const [features, setFeatures] = useState(Array(5).fill(""));

  const handleChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const result = await predictMushroom(features);
      onPrediction(result);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {features.map((feature, index) => (
        <div key={index}>
          <label>Feature {index + 1}</label>
          <input
            type="text"
            value={feature}
            onChange={(e) => handleChange(index, e.target.value)}
          />
        </div>
      ))}
      <button type="submit">Predict</button>
    </form>
  );
}

export default InputForm;
