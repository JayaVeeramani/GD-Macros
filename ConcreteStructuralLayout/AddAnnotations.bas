Attribute VB_Name = "AddAnnotations"



Function SelectAndAddAnnotation(swEnt As Object, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, SelXPos As Double, _
       SelYPos As Double, AnnXPos As Double, AnnYPos As Double, Optional CustomTextTop As String = "", _
        Optional CustomTextBottom As String = "") As SldWorks.Annotation

    Dim IsSelected As Boolean
    IsSelected = SelectEntityWithSelectData(swEnt, swView, swDrawing, SelXPos, SelYPos)
    
    If IsSelected Then

        Dim BalloonContent As swBalloonTextContent_e
    
        If CustomTextTop = "" And CustomTextBottom = "" Then
            BalloonContent = swBalloonTextPartNumberBOM
        End If
        
        Dim swNote As SldWorks.Note
        Set swNote = swDrawing.InsertBOMBalloon2(swBS_Inspection, swBF_Tightest, BalloonContent, _
                                CustomTextTop, BalloonContent, CustomTextBottom)
            
        If Not swNote Is Nothing Then
        
            Dim swAnnotation As SldWorks.Annotation
            Set swAnnotation = swNote.GetAnnotation()
        
            swAnnotation.SetPosition AnnXPos, AnnYPos, 0
    
            Set SelectAndAddAnnotation = swAnnotation
        
        End If

    End If
    
End Function

Function SelectAndAddItemNoAnnotation(swEnt As Object, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, SelXPos As Double, _
       SelYPos As Double, AnnXPos As Double, AnnYPos As Double, Optional IsLeaderReq As Boolean = False) As SldWorks.Annotation

    Dim IsSelected As Boolean
    IsSelected = SelectEntityWithSelectData(swEnt, swView, swDrawing, SelXPos, SelYPos)
    
    If IsSelected Then
    
        Dim swBalloonParams As SldWorks.BalloonOptions
        Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
        swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
        swBalloonParams.Style = swBalloonStyle_e.swBS_Circular
        swBalloonParams.UpperTextContent = swBalloonTextContent_e.swBalloonTextCutlistProperties
        swBalloonParams.UpperText = "$PRPWLD:" & Chr(34) & "ITEM NO" & Chr(34)

        Dim swNote As SldWorks.Note
        Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
    
        swNote.PropertyLinkedText = "$PRPWLD:" & Chr(34) & "ITEM NO" & Chr(34)
            
        If Not swNote Is Nothing Then
        
            Dim swAnnotation As SldWorks.Annotation
            Set swAnnotation = swNote.GetAnnotation()
        
            swAnnotation.SetPosition AnnXPos, AnnYPos, 0
            
            If IsLeaderReq Then
            
                Dim HeadStyle As Integer
                swAnnotation.SetLeader3 swLeaderStyle_e.swAlwaysAttachToBalloon + swLeaderStyle_e.swSTRAIGHT, swLeaderSide_e.swLS_SMART, False, False, True, False
                HeadStyle = swAnnotation.SetArrowHeadStyleAtIndex(0, swArrowStyle_e.swCLOSED_ARROWHEAD)
                
            Else
            
                swAnnotation.SetLeader3 swLeaderStyle_e.swNO_LEADER, swLeaderSide_e.swLS_SMART, False, False, True, False
            
            End If
    
            Set SelectAndAddItemNoAnnotation = swAnnotation
        
        End If

    End If
End Function

Function AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, yPos As Double) As SldWorks.Note
            
    Set AddNoteToView = swDrawing.InsertNote(NoteText)
            
    If Not AddNoteToView Is Nothing Then

        Dim swAnnotation As SldWorks.Annotation
        Set swAnnotation = AddNoteToView.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, yPos, 0

        End If

    End If
    
End Function

