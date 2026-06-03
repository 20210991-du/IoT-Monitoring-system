// 이미지 파일 → 살균된 data URL.
//  · 캔버스 재인코딩으로 EXIF/embedded 스크립트 제거(SVG 등 비-raster 는 type 체크로 거부) + 리사이즈.
//  · square=true 면 중앙 정사각 크롭(아바타용).
//  · 출력 mime 는 png/jpeg/webp — 서버 화이트리스트 regex 와 일치.
export function imageFileToDataURL(file, { max = 1024, square = false, mime = "image/jpeg", quality = 0.85 } = {}) {
  return new Promise((resolve, reject) => {
    if (!file || !/^image\/(png|jpe?g|webp)$/.test(file.type || "")) { reject(new Error("PNG·JPG·WEBP 이미지만 가능합니다.")); return; }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      try {
        let sw = img.naturalWidth, sh = img.naturalHeight, sx = 0, sy = 0;
        if (!sw || !sh) { reject(new Error("이미지를 읽을 수 없습니다.")); return; }
        if (square) { const s = Math.min(sw, sh); sx = (sw - s) / 2; sy = (sh - s) / 2; sw = sh = s; }
        const scale = Math.min(1, max / Math.max(sw, sh));
        const dw = Math.max(1, Math.round(sw * scale)), dh = Math.max(1, Math.round(sh * scale));
        const canvas = document.createElement("canvas");
        canvas.width = dw; canvas.height = dh;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, dw, dh);
        resolve(canvas.toDataURL(mime, quality));
      } catch (e) { reject(e); }
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("이미지 로드에 실패했습니다.")); };
    img.src = url;
  });
}
