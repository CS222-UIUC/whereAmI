const comments = [];

function renderComments() {
    event.preventDefault();
    const content = document.getElementById('contentOfTextarea').value;
    comments.push(content);

    let commentsHTML = '';
    
    comments.forEach((comment) => {
        commentsHTML += `
            <div class="comments">
                <strong class="username"> Anonymous </strong>
                <p class="usercomment">${comment}</p>
            </div>
        `;
    })
    console.log(commentsHTML);
    document.querySelector('.js-comment-section').
    innerHTML = commentsHTML;
    document.getElementById('contentOfTextarea').value = '';
}