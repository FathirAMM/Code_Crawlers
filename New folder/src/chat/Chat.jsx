import React from 'react'
import { Form, FormControl, Button, Image } from "react-bootstrap";
import { Tab, Tabs } from 'react-bootstrap';
import { ChatMessage } from './ChatMessage'
import { ChatSideBar } from './ChatSideBar'
import { ChatOpitons } from './ChatOpitons'
import { Apps } from './apps/index'
import { ChatList } from './ChatList'
import { classnames } from '../components/utils'
import { useGlobal } from './context'
import { Search } from '@/components'
import styles from './style/chat.module.less'
import { ScrollView } from './component'
import InjectToVectorDB from '../InjectToVectorDB';
//import ChatWithPDFs from '../ChatWithPdf';
//import ChatMessagePdf from './ChatMessagePdf';
import Finetunedd from '../finetune';
import { useOptions } from './hooks';
import { options } from 'less';
import { Icon } from '../components';
import { FileUploader } from './fileupload';

export default function Chats() {
  const { is } = useGlobal()
  const chatStyle = is.fullScreen ? styles.full : styles.normal
  // console.log(is);
  const { setGeneral } = useOptions()
  
  // setGeneral({ theme: options.general.theme === 'light' ? 'dark' : 'light' });
  // options.general.theme = 'dark';
  
  // console.log(options.general);
  const onSearch = (e) => {
    console.log(e)
  }
  return (
    <div id="x">
    <Tabs defaultActiveKey="rag" id="rag-tab">
    {/* <Tab eventKey="chat" title="Chat with Pdfs"> */}
      
      {/* <div className={classnames(styles.chat, chatStyle)}> */}

{/* <div className={styles.chat_inner}> */}

  {
    // is.config ?
  //   <React.Fragment>
  //   {
  //     is.sidebar && <div className={styles.sider}>
  //       <div className={styles.search}>
  //         <Search onSearch={onSearch} />
  //       </div>
  //       <ScrollView>
  //         {is.apps ? <Apps /> : <ChatList />}
  //       </ScrollView>
  //     </div>
  //   }
  //   {/* <ChatMessage /> */}
  // </React.Fragment> :
      // <React.Fragment>
      //   {
      //     is.sidebar && <div className={styles.sider}>
            
      //       <ScrollView>
      //         <Apps />
              
      //       </ScrollView>
      //       <div style={{padding:"5%"}}>
      //       <Form.Control id="test" type="file" label="Upload PDF" />
      //       <br />
            
      //       <Button >Submit</Button></div>
      //     </div>
      //   }
      //   <ChatMessagePdf />
      // </React.Fragment>
  }
{/* </div>
</div> */}
        {/* </Tab> */}
        <Tab eventKey="db" title="Inject to Vector DB">
         {/* <FileUploader/> */}
          <InjectToVectorDB />
          
        </Tab>
        <Tab eventKey="finetune" title="finetuning"> 
          <Finetunedd />
        </Tab>
      <Tab eventKey="rag" title="RAG Application">
      
    <div className={classnames(styles.chat, chatStyle)}>

      <div className={styles.chat_inner}>
      
        <ChatSideBar />
        {
          is.config ?
            <ChatOpitons /> :
            <React.Fragment>
              {
                is.sidebar && <div className={styles.sider}>
                  <div className={styles.search}>
                    <Search onSearch={onSearch} />
                  </div>
                  <ScrollView>
                    {is.apps ? <Apps /> : <ChatList />}
                  </ScrollView>
                </div>
              }
              <ChatMessage />
            </React.Fragment>
        }
      </div>
    </div>
    </Tab>
    <Tab eventKey="evaluvate" title="Evaluation">
    <div style={{background:'black'}}>
    <iframe
  src={'/plot2.html'}
  frameborder="100"
  style={{ width: '50%', height: '94vh', border: 'none' }}
></iframe>
<iframe
  src={'/plot1.html'}
  frameborder="100"
  style={{ width: '50%', height: '94vh', border: 'none' ,overflow:'scroll'}}
></iframe>
      
    </div>
    </Tab>
    </Tabs>
    </div>
  )
}
