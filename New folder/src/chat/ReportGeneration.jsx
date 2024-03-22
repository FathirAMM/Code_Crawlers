import React, { useState } from "react"
import { Button } from "../components/Button"
import { Input } from "../components"

const ReportGeneration = () =>{
    const [isGenerated,setIsGenerated] = useState(false)
    const [reportNumber,setReportNumber] = useState(0)
    const [reportStatus,setReportStatus] = useState("")

    const handleSubmit = async (reportNumber) => {
        setIsGenerated(false)
        setReportStatus("Proccessing")
        try {
          const response = await fetch('http://127.0.0.1:5000/api/report_generation', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              number: reportNumber,
            }),
          });
          setIsGenerated(true)
          if (response.ok) {
            setReportStatus('Success');
          } else {
            setReportStatus('Failed to Submit Report');
          }
        } catch (error) {
            console.log(error);
            setReportStatus('Failed to Submit Report');
        }
    }
    return <div style={{margin:"0px 10px"}}>
        <div style={{display:"flex",justifyContent:"space-between"}}>
            <Input name="reportNumber" onChange={(e1) => setReportNumber(e1)}/>
            <Button onClick={()=>handleSubmit(reportNumber)}>Generate</Button>
        </div>
       {isGenerated && reportStatus == "Success" && <div style={{display:"flex",justifyContent:"space-around",marginTop:"10px"}}>
            <Button>Download Word</Button>
            <Button>Download PDF</Button>
        </div>}
    </div>
}

export default ReportGeneration