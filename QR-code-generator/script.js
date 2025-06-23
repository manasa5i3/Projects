
 function generateqr(){
    let qrcode=document.getElementById("inputtag").value;
    let goourl="https://www.google.com/search?tbm=isch&q="+qrcode
    let res="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data="+encodeURIComponent(goourl);
    document.getElementById("imgtag").src= res;//safely encode a URL by converting special characters 
    // (like spaces, &, ?, =, etc.) into a format that browsers and APIs can understand correctly.
}


