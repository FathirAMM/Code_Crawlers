export async function getApiResponse(question, model, database,apikey) {
  const url = "http://127.0.0.1:5000/api/answer_question";
  try {
    console.log(question + " " + model + " " + database+" "+apikey);
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        model: model,
        database: database,
        apikey:apikey,
      }),
    });
    console.log(response);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const responseData = await response.json();
    console.log(typeof responseData.result);
    return responseData;
  } catch (error) {
    console.error("Error fetching API:", error);
    return null;
  }
}

export async function getInjectToVectorDBApiResponse(question, model, database) {
  const url = "http://127.0.0.1:5000/api/inject_to_vector_db";
  try {
    console.log(question + " " + model + " " + database);
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        model: model,
        database: database,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const responseData = await response.json();
    console.log(typeof responseData.result);
    return responseData;
  } catch (error) {
    console.error("Error fetching API:", error);
    return null;
  }
}

