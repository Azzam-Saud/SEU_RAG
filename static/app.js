async function send() {
  const input = document.getElementById("question");
  const chat = document.getElementById("chat-box");

  if (!input.value) return;

  chat.innerHTML += `<div class="msg user">👤 ${input.value}</div>`;

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: input.value })
  });

  const data = await res.json();

  chat.innerHTML += `<div class="msg bot">🤖 ${data.answer}</div>`;
  chat.scrollTop = chat.scrollHeight;

  input.value = "";
}