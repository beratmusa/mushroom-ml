export async function predictMushroom(features) {
  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ features }),
  });

  if (!response.ok) {
    throw new Error("Prediction API call failed");
  }

  return response.json();
}
