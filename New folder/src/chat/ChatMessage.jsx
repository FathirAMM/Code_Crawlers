import React, { useState } from 'react'
import { Avatar, Icon, Textarea, Loading, Tooltip, Button, Popover } from '@/components'
import { CopyIcon, ScrollView, Error, EmptyChat, ChatHelp } from './component'
import { MessageRender } from './MessageRender'
import { ConfigInfo } from './ConfigInfo'
import { useGlobal } from './context'
import { useMesssage, useSendKey, useOptions } from './hooks'
import { dateFormat } from './utils'
import avatar from '@/assets/images/avatar-gpt.png'
import styles from './style/message.module.less'
import { classnames } from '../components/utils'
import { stt } from './service/stt'
import Feedback from './Feedback'
import ReportButton from './ReportButton'

export function MessageHeader() {
  const { is, setIs, clearMessage, options } = useGlobal()
  const { message } = useMesssage()
  const { messages = [] } = message || {}
  const columnIcon = is.sidebar ? 'column-close' : 'column-open'
  const { setGeneral } = useOptions()
  const { typemessage } = useGlobal();
  const x = document.getElementById('x');
  const downloads = () =>{};
  return (
    <div className={classnames(styles.header)}>
      <Button type="icon" icon={columnIcon} onClick={() => setIs({ sidebar: !is.sidebar })} />
      <div className={styles.header_title}>
        {message?.title}
        <div className={styles.length}>{messages.length} messages</div>
      </div>
      <div className={styles.header_bar}>
        <Icon className={styles.icon} type={options.general.theme} onClick={() => setGeneral({ theme: options.general.theme === 'light' ? 'dark' : 'light' })} />
        
        <Icon className={styles.icon} type="clear" onClick={clearMessage} />
        {/* <Popover position="bottom" content={<ConfigInfo />}>
          <Icon className={styles.icon} type="more" />
        </Popover> */}
        <Icon type="download" className={styles.icon} />
      </div>
    </div>
  )
}

export function EditorMessage() {
  return (
    <div>
      <Textarea rows="3" />
    </div>)
}

export function MessageItem(props) {
  const { content, sentTime, role, images,id, requested,url} = props
  const [isFeedback,setIsFeedback] = useState(false)
  const { removeMessage } = useGlobal()
  // console.log(content);
  // console.log(requested)
// console.log(images);
  // Check if images is an array before mapping over it
  if(url != null){
    console.log(url);
  }
  return (
    
    <div className={classnames(styles.item, styles[role])}>
      <Avatar src={role !== 'user' && avatar} />
      <div className={classnames(styles.item_content, styles[`item_${role}`])}>
      <div>
        <div className={styles.item_inner}>
          <div className={styles.item_tool}>
            <div className={styles.item_date}>{dateFormat(sentTime)}</div>
            <div className={styles.item_bar}>
            {role !== 'user' &&  <ReportButton messageId={id}  responsed={content}/>}
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
          <MessageRender>{content}</MessageRender>
          <img width="40%"src={images??""}/>
          <div>
          {role !== 'user' &&  <Icon className={styles.icon} type="editor" onClick={() =>setIsFeedback(true)} />}
          </div>
          {
          isFeedback && <Feedback messageId={id} setIsFeedback={setIsFeedback} message={requested} responsed={content}/>

        }
        
        </div>
        {role !== 'user' && 
        <div className={classnames(styles.item, styles[role])}>
      
      <div className={classnames(styles.item_content, styles[`item_${role}`])}>
        <div className={styles.item_inner} style={{display:"flex",flexDirection:'row',gap:"1rem"}}>
       
        {url!= null && url.map((item) => <a href={item}> {role !== 'user' && item}</a>)}
      </div></div></div>}
      </div>
      </div>
    </div>
  )
}


export function MessageBar() {
  const { sendMessage, setMessage,clearTypeing,is, options, setIs, typeingMessage,  stopResonse } = useGlobal()
  useSendKey(sendMessage, options.general.command)
//   const clearTypeing = () => {
//       console.log('test');
//     document.getElementById("textarea").value = '';
// };
  return (
    <div className={styles.bar}>
      {is.thinking && <div className={styles.bar_tool}>
        <div className={styles.bar_loading}>
          <div className="flex-c"><span>Thinking</span> <Loading /></div><Button size="min" className={styles.stop} onClick={stopResonse} icon="stop">Stop Response</Button>
        </div>
      </div>}
      <div className={styles.bar_inner}>
        <div className={styles.bar_type}>
          <Textarea id="textarea1" transparent={true} rows="3" value={typeingMessage?.content || ''} onFocus={() => setIs({ inputing: true })} onBlur={() => setIs({ inputing: false })} placeholder="Enter something...." onChange={setMessage} />
        </div>
        <div className={styles.bar_icon}>
          {/* {typeingMessage?.content &&
            <Tooltip text="clear">
              <Icon className={styles.icon} type="cancel" onClick={clearTypeing} />
            </Tooltip>} */}
          {/* <Tooltip text="history">
            <Icon className={styles.icon} type="history" />
          </Tooltip> */}
          <Icon className={styles.icon} type="send" onClick={sendMessage} />
          <Icon className={styles.icon} type="mic" onClick={stt} />
        </div>
      </div>
    </div>
  )
}

export function MessageContainer() {
  const { options } = useGlobal()
  const { message } = useMesssage()
  const { messages = [] } = message || {}
  if (options?.openai?.apiKey) {
    return (
      <React.Fragment>
        {
          messages.length ? <div className={styles.container}>
            {messages.map((item, index) => <MessageItem key={index} {...item} />)}
            
            {/* {message?.error && <Error />} */}
          </div> : <ChatHelp />
        }
      </React.Fragment>
    )
  } else {
    return <EmptyChat />
  }
}

// Your existing code
export function ChatMessage() {
  const { is } = useGlobal()
  return (
    <React.Fragment>
      <div className={styles.message}>
        <MessageHeader />
        <ScrollView>
          <MessageContainer />
          {is.thinking && <Loading />}
        </ScrollView>
        <MessageBar />
      </div>
    </React.Fragment >
  )
}