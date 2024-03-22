import React from 'react'
import styles from './style.module'
import EmptyImg from './EmptyImg'
import './ChatHelp.css';
import action from '../context/action';
import { Button } from 'react-bootstrap';
import { ChatMessage,MessageBar } from '../ChatMessage';

export function ChatHelp() {
  const handleButtonClick = () => {
    // Call the triggerSendButtonClick function from the ChatHelp component
    
  };
  
  return (
    <div className={styles.help}>
      <h1 className='name'>GPT-Chat</h1>
      <div className='caption-rag'>Your trusty sidekick for all things organization</div><br /><br />
      <div className='tellcontainer'>
        <div className='tells'>
          <div className='tell' >
            <div>
              <img width="96px" src="./economy.png" alt="economy" />
            </div>
            <div>
              Economy
              <p>main causes of inflation in Sri Lanka</p>
            </div>
            {/* <Button size='min' className={styles.stop} onClick={handleButtonClick}>click</Button> */}
            </div>
            <div className='tell'>
            <div>
              <img width="96px" src="./inflation.png" alt="economy" />
            </div>
            <div>
              Fluctuated
              <p>how did money multiplier change between 2021 and 2022</p>
            </div>
            </div>
          </div>
          <div className='tells'>
            <div className='tell'>
              <div>
              <img width="96px" src="./economy growth.png" alt="economy" />
              </div>
              <div>
            Growth
            <p>major contributors of the manufacturing sector in 2022</p>
            </div></div>
            <div className='tell'>
            <div>
              <img width="96px" src="./unemployment rate.png" alt="economy" />
              </div>
              <div>
            Improvement
            <p>Why did the female unemployment rate decline in 2022?</p>
            </div>
            </div>
          </div>
        </div>
    </div>
  )
}
