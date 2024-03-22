import React, { useState } from "react"
import { Avatar, Icon, Textarea, Loading, Tooltip, Button, Popover } from '@/components'
const feedbacks = ["Good","Average","Not factually correct","Didn't follow instructions"]
const handleSubmit = async (messageId,message,feedback,responsed) => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/submit_feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: messageId,
          message: message,
          feedback: feedback,
          response: responsed,
        }),
      });

      if (response.ok) {
        console.log('Feedback submitted successfully');
      } else {
        console.error('Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
}
const Feedback = ({setIsFeedback,messageId,message,responsed}) => {
    const [isSubmitted,setIsSubmitted] = useState(false)
    const onFeedbackHandle = (feedback) =>{
        console.log(feedback)
        console.log(messageId)
        console.log(message)
        console.log(responsed)
        setIsSubmitted(true)
        handleSubmit(messageId,message,feedback,responsed);
    }
    return <div>
        <Icon type="close" onClick={() =>setIsFeedback(false)}>Close</Icon>
        <h6><strong>feedback</strong></h6>
        {
            !isSubmitted ? <div style={{display:"flex", direction:"row",gap:"1rem"}}>
                {feedbacks.map((feedback,index) => <div ><button key={index} className="button-black" style={{gap:"0.75rem"}} onClick={() => onFeedbackHandle(feedback)}>{feedback}&ensp;</button><br /></div>)}
            </div>:<p>Feedback Submitted</p>
            
        }
    </div>
}

export default Feedback