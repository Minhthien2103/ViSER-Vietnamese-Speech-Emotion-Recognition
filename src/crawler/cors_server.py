import http.server
import socketserver

PORT = 8081

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Đây chính là "tấm vé thông hành" để trình duyệt không chặn nữa
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
    print(f"📡 Trạm phát sóng có hỗ trợ CORS đang chạy tại cổng {PORT}...")
    httpd.serve_forever()