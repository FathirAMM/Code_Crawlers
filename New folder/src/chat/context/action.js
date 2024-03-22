import React from "react";
import { getApiResponse } from "../service/openai.mjs";
// $contents = getApiResponse();


// export async function chatWithPDF(question, file) {
//   const url = "http://127.0.0.1:5000/api/chat_with_pdfs";
//   const formData = new FormData();
//   formData.append('question', question);
//   formData.append('file', file);

//   try {
//     const response = await fetch(url, {
//       method: "POST",
//       body: formData, // Note: When using FormData, don't set Content-Type header
//     });
//     if (!response.ok) {
//       throw new Error(`HTTP error! Status: ${response.status}`);
//     }
//     const responseData = await response.json();
//     return responseData;
//   } catch (error) {
//     console.error("Error in chatWithPDF:", error);
//     return null;
//   }
// }

export function MessageItem(props) {
  const { content, sentTime, role, images,url } = props
  const { removeMessage } = useGlobal()
  if(url != null){
    console.log(url);
  }
// console.log(images);
  // Check if images is an array before mapping over it
  return (
    <div className={classnames(styles.item, styles[role])}>
      <Avatar src={role !== 'user' && avatar} />
      <div className={classnames(styles.item_content, styles[`item_${role}`])}>
        <div className={styles.item_inner}>
          <div className={styles.item_tool}>
            <div className={styles.item_date}>{dateFormat(sentTime)}</div>
            <div className={styles.item_bar}>
              <Tooltip text="Remove Messages">
                <Icon className={styles.icon} type="trash" onClick={removeMessage} />
              </Tooltip>
              {role === 'user' ? (
                <>
                  <Icon className={styles.icon} type="reload" />
                  <Icon className={styles.icon} type="editor" />
                </>
              ) : (
                <CopyIcon value={content} />
              )}
            </div>
          </div>
          {content}
          <img width="20%"src={images}/>
        </div>
        
      </div>
    </div>
  )
}

export default function action(state, dispatch) {
  // console.log(state.options.openai.database);
  const setState = (payload = {}) =>
    dispatch({
      type: "SET_STATE",
      payload: { ...payload },
    });
    
  return {
    setState,
    async clearTypeing() {
      document.getElementById("textarea1").value = '';
      console.log(state.typeingMessage.content);
      state.typeingMessage.content = "";
      console.log(state.typeingMessage);
    },
    async sendMessage() {
      const { typeingMessage, chat, currentChat } = state;
      if (typeingMessage?.content) {
        const newMessage = {
          ...typeingMessage,
          sentTime: Date.now(),
        };
        try {
          setState({
            is: { ...state.is, thinking: true }, // Set thinking to true
          });

          const { result, relevant_images,urls } = await getApiResponse(
            typeingMessage.content,
            state.options.openai.model,
            state.options.openai.database,
            state.options.openai.apiKey// Pass the model value from state.datas to the API request
          );
          if(relevant_images!==null){
          const imageData = relevant_images[0];
          const imageHTML = imageData!=null?`data:image/jpeg;base64,${imageData}`:"";

          const fullMessageContent = `${imageHTML}`;

          const helloMessage = {
            images: fullMessageContent,
            content: result, // Assuming result is a string
            role: "assistant",
            sentTime: Date.now() + 1, // Ensure the hello message appears after the user's message
            id: Date.now() + 1,
            requested:typeingMessage.content,
            url:urls,
          };
        

          const messages = [
            ...chat[currentChat].messages,
            newMessage,
            helloMessage,
          ];

          let newChat = [...chat];
          newChat.splice(currentChat, 1, { ...chat[currentChat], messages });
          state.is.typeing = false;
          console.log(state.is.typeing);
          
          setState({
            typeingMessage: {},
            chat: newChat,
            is: { ...state.is, thinking: false, typeing:{}}, // Set thinking back to false
          });
          
          // Display relevant_images as images
          
        }
        else{
          const helloMessage = {
            
            content: result, // Assuming result is a string
            role: "assistant",
            sentTime: Date.now() + 1, // Ensure the hello message appears after the user's message
            id: Date.now() + 1,
            url:urls,
          };
        

          const messages = [
            ...chat[currentChat].messages,
            newMessage,
            helloMessage,
          ];

          let newChat = [...chat];
          newChat.splice(currentChat, 1, { ...chat[currentChat], messages });
          setState({
            typeingMessage: "",
            chat: newChat,
            is: { ...state.is, thinking: false }, 
            // Set thinking back to false
          });
          
        }
        } catch (error) {
          console.error("Error fetching API:", error);
          setState({
            is: { ...state.is, thinking: false }, // Set thinking back to false in case of error
          });
        }
      }
      
      console.log(document.getElementById("textarea1").value);
    },
    async sendMessaged() {
      const question = document.getElementById("textarea").value??0;
      
      try{
      document.getElementById("messagecontainer").innerHTML += "<div class='item_content-d32cf'><div class='item_inner-a6eef'><div class='item_tool-b7994'><div class='item_date-d1450'>21.00</div><div class='item_bar-e575f'><div class='tooltip-d272c'><i class='icon-a3291 ico ico-trash icon-e54d3'></i><div class='container-bb450 top-b801a'><div class='inner-d4680'>Remove Messages</div></div></div><i class='icon-a3291 ico ico-reload icon-e54d3'></i><i class='icon-a3291 ico ico-editor icon-e54d3'></i></div></div><div class='z-ui-markdown'><p>HELLO</p></div><img width='20%'></div></div></div>";
      }
      catch(e){
        console.log(e);
      }
      // console.log(question)
       const file =  document.getElementById("test").files[0]??0;
      const { typeingMessage, chat, currentChat } = state;
      if (typeingMessage?.content) {
        const newMessaged = {
          ...typeingMessage,
          sentTime: Date.now(),
        };
        try {
          setState({
            is: { ...state.is, thinking: true }, // Set thinking to true
          });

          //const responseData = await chatWithPDF(question, file);
          // console.log(responseData);
          // const messageItemComponent = <MessageItem content={responseData.answer} sentTime={ Date.now() + 1} role="assistant" images="null" />;
          $date = Date.now +1;
          // console.log($date);
          // console.log(responseData.answer);
          document.getElementById("messagecontainer").innerHTML += "<div class='item_content-d32cf'><div class='item_inner-a6eef'><div class='item_tool-b7994'><div class='item_date-d1450'>"+$date+"</div><div class='item_bar-e575f'><div class='tooltip-d272c'><i class='icon-a3291 ico ico-trash icon-e54d3'></i><div class='container-bb450 top-b801a'><div class='inner-d4680'>Remove Messages</div></div></div><i class='icon-a3291 ico ico-reload icon-e54d3'></i><i class='icon-a3291 ico ico-editor icon-e54d3'></i></div></div><div class='z-ui-markdown'><p"+responseData.answer+"</p></div><img width='20%'></div></div></div>";
          setState({
            typeingMessage: {},
            is: { ...state.is, thinking: false,typeingMessage }, 
            // Set thinking back to false
          });
          console.log(state);
        // }
        } catch (error) {
          console.error("Error fetching API:", error);
          setState({
            is: { ...state.is, thinking: false }, // Set thinking back to false in case of error
          });
        }
      }
    },

    newChat() {
      const { chat } = state;
      const chatList = [
        ...chat,
        {
          title: "This is a New Conversations",
          id: Date.now(),
          messages: [],
          ct: Date.now(),
          icon: [2, "files"],
        },
      ];
      setState({ chat: chatList, currentChat: chatList.length - 1 });
    },

    modifyChat(arg, index) {
      const chat = [...state.chat];
      chat.splice(index, 1, { ...chat[index], ...arg });
      setState({ chat, currentEditor: null });
    },

    editChat(index, title) {
      const chat = [...state.chat];
      chat.splice(index, 1, [...chat[index], title]);
      setState({
        chat,
      });
    },
    removeChat(index) {
      const chat = [...state.chat];
      chat.splice(index, 1);
      const payload =
        state.currentChat === index
          ? { chat, currentChat: index - 1 }
          : { chat };
      setState({
        ...payload,
      });
    },

    async setMessage(content) {
      const typeingMessage =
       await content === ""
          ? {}
          : {
              role: "user",
              content,
              id: Date.now(),
            };
      setState({ is: { ...state.is, typeing: true }, typeingMessage });
      
      
     
    },

    clearMessage() {
      const chat = [...state.chat];
      chat[state.currentChat].messages = [];
      setState({
        chat,
      });
    },

    removeMessage(index) {
      const messages = state.chat[state.currentChat].messages;
      const chat = [...state.chat];
      messages.splice(index, 1);
      chat[state.currentChat].messages = messages;
      setState({
        chat,
      });
    },

    setOptions({ type, data = {} }) {
      console.log(data.theme);
      if(data.theme=='dark'){
        document.querySelector('body').style.background = "#353232";
        document.getElementById('containt').style.background = "black";
        document.getElementById('containt1').style.background = "black";
        document.getElementsByClassName('container')[0].style.background = "black";
        document.getElementsByClassName('container')[1].style.background = "black";
        document.getElementsByClassName('upload-container')[0].style.background = "black";
        document.getElementsByClassName('upload-container')[1].style.background = "black";
        document.getElementsByClassName('border-container')[0].style.border = "5px solid rgba(198, 198, 198, 0.65)";
        document.getElementsByClassName('border-container')[1].style.border = "5px solid rgba(198, 198, 198, 0.65)";
        if(document.getElementsByClassName('name').length !==0 ){
        document.getElementsByClassName('name')[0].style.color = "white";
        }
        if(document.getElementsByClassName('bi').length !== 0){
          for (let index = 0; index < document.getElementsByClassName('bi').length; index++) {
            document.getElementsByClassName('bi')[index].style.fill = "white"; 
            document.getElementsByClassName('bi')[index].style.color = "white"; 
          }
        }
        if(document.getElementsByClassName('tell').length !== 0){
          for (let index = 0; index < document.getElementsByClassName('tell').length; index++) {
            document.getElementsByClassName('tell')[index].style.background = "rgb(37 41 49)";
            document.getElementsByClassName("tell")[index].style.color = "white";  
          }
        }
        
        }
      if(data.theme=='light'){
        document.querySelector('body').style.background = "white";
        document.getElementById('containt').style.background = "#f9f9f9";
        document.getElementById('containt1').style.background = "#f9f9f9";
        document.getElementsByClassName('container')[0].style.background = "#f9f9f9";
        document.getElementsByClassName('container')[1].style.background = "#f9f9f9";
        document.getElementsByClassName('upload-container')[0].style.background = "#f9f9f9";
        document.getElementsByClassName('upload-container')[1].style.background = "#f9f9f9";
        document.getElementsByClassName('border-container')[0].style.border = "5px dashed rgba(198, 198, 198, 0.65)";
        document.getElementsByClassName('border-container')[1].style.border = "5px dashed rgba(198, 198, 198, 0.65)";
        if(document.getElementsByClassName('name').length !==0 ){
        document.getElementsByClassName('name')[0].style.color = "black";
        }
        if(document.getElementsByClassName('bi').length !== 0){
          for (let index = 0; index < document.getElementsByClassName('bi').length; index++) {
            document.getElementsByClassName('bi')[index].style.fill = "rgb(37 41 49)";
            document.getElementsByClassName('bi')[index].style.fill = "rgb(37 41 49)"; 
              
          }
        }
        if(document.getElementsByClassName('tell').length !== 0){
          for (let index = 0; index < document.getElementsByClassName('tell').length; index++) {
            document.getElementsByClassName('tell')[index].style.background = "white";
            document.getElementsByClassName("tell")[index].style.color = "gray";  
          }
        }
        
      }
      let options = { ...state.options };
      options[type] = { ...options[type], ...data };
      setState({ options });
    },

    setIs(arg) {
      const { is } = state;
      setState({ is: { ...is, ...arg } });
    },

    currentList() {
      return state.chat[state.currentChat];
    },

    stopResonse() {
      setState({
        is: { ...state.is, thinking: false },
      });
    },
  };
}

export const datas = {
  id: "chatcmpl-7AEK9Dlw96m5TejBKIKUgjyUHVTCa",
  object: "chat.completion",
  created: 1682672697,
  model: "gpt-3.5-turbo-0301",
  usage: {
    prompt_tokens: 34,
    completion_tokens: 303,
    total_tokens: 337,
  },
  choices: [
    {
      message: {
        role: "assistant",
        content: "test",
      },
      finish_reason: "stop",
      index: 0,
    },
  ],
};
