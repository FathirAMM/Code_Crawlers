import React, { useState } from "react"
import { Button } from "../components/Button"

const ReportButton = ({messageId,responsed}) =>{
    const [isReported,setIsReported] = useState(false)
    const [reportStatus,setReportStatus] = useState("Hello")

    const handleSubmit = async (messageId,responsed) => {
        try {
          const response = await fetch('http://127.0.0.1:5000/api/submit_report', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message_id: messageId,
              response: responsed,
            }),
          });
        setIsReported(true)
          if (response.ok) {
            setReportStatus('Report Submitted Successfully');
          } else {
            setReportStatus('Failed to Submit Report');
          }
        } catch (error) {
            setReportStatus('Failed to Submit Report');
            console.error('Error Submitting Report:', error);
        }
    }

    return <>
   {!isReported ?
    <Button onClick={()=>handleSubmit(messageId,responsed)}>Add Report</Button>
   :
   <span>{reportStatus}</span>}
    </>
}
export default ReportButton