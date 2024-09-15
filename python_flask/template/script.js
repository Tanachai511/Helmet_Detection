function changeImage(newSrc) {
    document.getElementById("dynamic-image").src = newSrc;
}

// เพิ่ม event listener ให้กับกล่องอัปโหลดรูปภาพ
document.getElementById("upload-box").addEventListener("mouseover", function() {
    changeImage('images/Screenshot 2024-07-30 235159.png'); // รูปใหม่ที่จะเปลี่ยนเมื่อ hover
});


// เพิ่ม event listener ให้กับกล่อง AI ตรวจจับ
document.getElementById("ai-detect-box").addEventListener("mouseover", function() {
    changeImage('images/home.png'); // รูปใหม่เมื่อ hover AI box
});
