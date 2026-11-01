/* The review page's /api requests are answered in the page itself (hosted/runtime.js). One
 * that reaches the network means the page was opened without a video chosen: say where to go. */
export async function onRequest() {
  return new Response(JSON.stringify({ detail: "Sign in first: open /start.html." }), {
    status: 409, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
