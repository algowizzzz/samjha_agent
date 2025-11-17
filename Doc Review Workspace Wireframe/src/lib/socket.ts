let socketReady: Promise<any> | null = null;

function injectSocketScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-sock-io="${src}"]`) as HTMLScriptElement | null;
    if (existing) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.defer = true;
    script.setAttribute('data-sock-io', src);
    script.onload = () => resolve();
    script.onerror = (e) => reject(e);
    document.head.appendChild(script);
  });
}

export async function getSocketClient() {
  if (!socketReady) {
    // Load client from CDN to avoid dev-proxy path issues
    socketReady = injectSocketScript('https://cdn.socket.io/4.7.5/socket.io.min.js').then(() => {
      // @ts-ignore
      return (window as any).io;
    });
  }
  return socketReady;
}

export async function connectSocket(token: string) {
  const io = await getSocketClient();
  // @ts-ignore
  const socket = io('http://127.0.0.1:8000', {
    transports: ['websocket'],
    withCredentials: true,
    auth: { token },
    path: '/socket.io',
  });
  return socket;
}

