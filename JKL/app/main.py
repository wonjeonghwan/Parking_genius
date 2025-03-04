from fastapi.responses import FileResponse, HTMLResponse
from fastapi import FastAPI
from routers.video import video_router

app = FastAPI()

app.include_router(video_router)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def main_page():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>주차 공간 분석</title>
        <style>
            body {{ text-align: center; margin-top: 50px; }}
            video, img {{ width: 80%; max-width: 960px; margin-top: 20px; }}
            .controls {{ margin-top: 20px; }}
        </style>
    </head>
    <body>

    <h2>주차 공간 분석</h2>

    <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">업로드 및 분석 시작</button>
    </form>

    <br>

    <h3>1초 프레임을 클릭하여 주차 공간 선택</h3>
    <img id="previewImage" src="" style="cursor:pointer; display:none;" onclick="sendClick(event)">

    <h3>처리된 영상 다운로드</h3>
    <a id="downloadLink" href="" download>
        <button id="downloadButton" style="display:none;">MP4 파일 다운로드</button>
    </a>

    <script>
        let videoId = "";

        document.querySelector("#uploadForm").onsubmit = async (e) => {{
            e.preventDefault();
            const formData = new FormData(e.target);

            const response = await fetch("/video/upload/", {{
                method: "POST",
                body: formData
            }});

            if (!response.ok) {{
                alert("🚨 영상 업로드 실패!");
                return;
            }}

            const result = await response.json();
            videoId = result.video_id;
            document.getElementById("previewImage").src = `${{window.location.origin}}${{result.preview_url}}`;
            document.getElementById("previewImage").style.display = "block";
        }};

        async function sendClick(event) {{
            if (!videoId) {{
                alert("🚨 오류: 비디오 ID가 없습니다.");
                return;
            }}

            const rect = event.target.getBoundingClientRect();
            const x = Math.round(event.clientX - rect.left);
            const y = Math.round(event.clientY - rect.top);

            const response = await fetch("/video/select_parking_spot/", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json"
                }},
                body: JSON.stringify({{ "video_id": videoId, "x": x, "y": y }})
            }});

            if (!response.ok) {{
                alert("🚨 서버 응답 오류");
                return;
            }}

            const processedResult = await response.json();
            console.log("✅ 다운로드 URL:", processedResult.download_url);

            if (!processedResult.download_url) {{
                alert("🚨 영상 처리가 완료되지 않았습니다.");
                return;
            }}

            // ✅ 다운로드 버튼 설정
            document.getElementById("downloadLink").href = processedResult.download_url;
            document.getElementById("downloadButton").style.display = "inline-block";

            // ✅ 자동으로 다운로드 시작
            window.location.href = processedResult.download_url;
        }};
    </script>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)