Attribute VB_Name = "AddAnnotations"
Dim AllowableRebarGap As Double

Function AddRebarAnnotations(Dict As Scripting.Dictionary, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            DimDict As Scripting.Dictionary, TopDimKey As String, BottomDimKey As String, LeftDimKey As String, _
                RightDimKey As String)

    If Dict.Count > 0 Then
    
        Dim vArrListKeys As Variant
        vArrListKeys = Dict.Keys
        
        Dim i As Integer
        For i = LBound(vArrListKeys) To UBound(vArrListKeys)
        
            Dim ArrList As IArrListObject
            Set ArrList = Dict.Item(vArrListKeys(i))

            AllowableRebarGap = swView.ScaleDecimal * 3 * 0.0254 + 0.00875
            
            Dim vRebars As Variant
            vRebars = ArrList.Items
            
            Dim SelXPos As Double
            Dim SelYPos As Double
            Dim AnnXPos As Double
            Dim AnnYPos As Double
            
            Dim SelFace As SldWorks.Face2

            Dim IsLowerSide As Boolean
            Dim Percent As Double
            Percent = 0.25
            
            Dim ClearanaceX As Double
            Dim ClearanaceY As Double
            
            Dim j As Integer
            For j = LBound(vRebars) To UBound(vRebars)
            
                ClearanaceX = 0.005
                ClearanaceY = 0.0025
                
                Dim oRebar As IRebarBody
                Set oRebar = vRebars(j)
                
                If i = LBound(vArrListKeys) Then

                    Call GetEndRebarCallOutPositionInfo(oRebar, DimDict, swView, TopDimKey, BottomDimKey, LeftDimKey, _
                                RightDimKey, SelXPos, SelYPos, AnnXPos, AnnYPos, True, False, ClearanaceX, ClearanaceY)

                ElseIf i = UBound(vArrListKeys) Then
                
                    Call GetEndRebarCallOutPositionInfo(oRebar, DimDict, swView, TopDimKey, BottomDimKey, LeftDimKey, _
                                RightDimKey, SelXPos, SelYPos, AnnXPos, AnnYPos, False, False, ClearanaceX, ClearanaceY)

                Else

                    If vArrListKeys(i + 1) - vArrListKeys(i) < AllowableRebarGap Then
                        
                        If vArrListKeys(i) - vArrListKeys(i - 1) < AllowableRebarGap Then
                            
                            Dim PrevItems As Variant
                            PrevItems = Dict.Item(vArrListKeys(i - 1)).Items
                            
                            Dim PrevRebar As IRebarBody
                            Set PrevRebar = PrevItems(0)
                            
                            Call GetClearanceForRebar(oRebar, PrevRebar, ClearanaceX, ClearanaceY)
                            
                            If Percent <= 0.5 Then
                                
                                Percent = Percent + 0.25
                                    
                            Else
                                
                                Percent = 0.25
                                    
                            End If
                        
                        End If
                        
                        IsLowerSide = True
                        
                    Else
                    
                        IsLowerSide = False
                        
                    End If
                    
                    Call GetEndRebarCallOutPositionInfo(oRebar, DimDict, swView, TopDimKey, BottomDimKey, LeftDimKey, _
                                RightDimKey, SelXPos, SelYPos, AnnXPos, AnnYPos, IsLowerSide, True, ClearanaceX, ClearanaceY, Percent)
                
                End If
                
                Set SelFace = GetLargestVisibleFaceCorrespondingToABody(oRebar.GetComponent, oRebar.GetBody, swView)
                Call SelectAndAddItemNoAnnotation(SelFace, swDrawing, swView, SelXPos, SelYPos, AnnXPos, AnnYPos, True)

            Next j

        Next i

    End If
    
End Function

Sub GetClearanceForRebar(oRebar As IRebarBody, PrevRebar As IRebarBody, ByRef ClearanceX As Double, ByRef ClearanceY As Double)

    If oRebar.IsHorizontal Then
    
        ClearanceY = Abs(oRebar.yMinSketchPoint - PrevRebar.yMinSketchPoint) - 0.00175

    Else
    
        ClearanceX = Abs(oRebar.xMinSketchPoint - PrevRebar.xMinSketchPoint) - 0.00175
        
    End If

End Sub

Sub GetEndRebarCallOutPositionInfo(oRebar As IRebarBody, DimDict As Scripting.Dictionary, swView As SldWorks.View, _
            TopDimKey As String, BottomDimKey As String, LeftDimKey As String, RightDimKey As String, ByRef SelXPos As Double, _
            ByRef SelYPos As Double, ByRef AnnXPos As Double, ByRef AnnYPos As Double, IsLowerRebar As Boolean, _
            AddAllDims As Boolean, ClearanaceX As Double, ClearanaceY As Double, Optional Percent As Double = 0.25)   '

    Dim KeyVal As String
    Dim DimArrList As IArrList
    Dim AdditionalArrList As IArrList
    Dim DimKey As String
    
    Dim GapMin As Double
    Dim GapMax As Double
    
    If oRebar.IsHorizontal Then
        
        If IsLowerRebar Then
            
            SelYPos = oRebar.yMin
            AnnYPos = SelYPos - AllowableRebarGap + ClearanaceY - 0.0025
            Set DimArrList = GetDimDictDataWithKey(DimDict, BottomDimKey)
            Set AdditionalArrList = GetDimDictDataWithKey(DimDict, TopDimKey)
            
        Else
        
            SelYPos = oRebar.yMax
            AnnYPos = SelYPos + AllowableRebarGap - ClearanaceY
            Set DimArrList = GetDimDictDataWithKey(DimDict, TopDimKey)
            Set AdditionalArrList = GetDimDictDataWithKey(DimDict, BottomDimKey)
 
        End If
        
        If AddAllDims Then
        
            DimArrList.AddItems AdditionalArrList.Items
            
        End If
        
        Call GetPositionWithMaxGap(DimArrList, GapMin, GapMax, oRebar, "xMin", "xMax")

        SelXPos = GapMin + Percent * (GapMax - GapMin)
        AnnXPos = SelXPos - 0.002

    Else
    
        If IsLowerRebar Then
        
            SelXPos = oRebar.xMin
            AnnXPos = SelXPos - AllowableRebarGap + ClearanaceX - 0.0025
            Set DimArrList = GetDimDictDataWithKey(DimDict, LeftDimKey)
            Set AdditionalArrList = GetDimDictDataWithKey(DimDict, RightDimKey)
            
        Else
        
            SelXPos = oRebar.xMax
            AnnXPos = SelXPos + AllowableRebarGap - ClearanaceX
            Set DimArrList = GetDimDictDataWithKey(DimDict, RightDimKey)
            Set AdditionalArrList = GetDimDictDataWithKey(DimDict, LeftDimKey)

        End If
        
        If AddAllDims Then
        
            Call DimArrList.AddItems(AdditionalArrList.Items)
            
        End If
        
        Call GetPositionWithMaxGap(DimArrList, GapMin, GapMax, oRebar, "yMin", "yMax")
        SelYPos = GapMin + Percent * (GapMax - GapMin)
        AnnYPos = SelYPos + 0.00875
    
    End If


End Sub


Function GetLargestVisibleFaceCorrespondingToABody(swComp As SldWorks.Component2, swBody As SldWorks.Body2, swView As SldWorks.View)

    Dim vEnts As Variant
    vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Face)

    If Not IsEmpty(vEnts) Then
        
        Dim Area As Double
        Area = 0
        
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swFace As SldWorks.Face2
            Set swFace = vEnts(i)
            
            Dim swFaceBody As SldWorks.Body2
            Set swFaceBody = swFace.GetBody
                
            If swFaceBody.Name = swBody.Name Then
                
                If swFace.GetArea > Area Then
                    
                    Set GetLargestVisibleFaceCorrespondingToABody = swFace
                    Area = swFace.GetArea
                    
                End If

            End If
            
        Next i

    End If

End Function


Sub GetPositionWithMaxGap(ArrList As IArrList, ByRef GapMin As Double, ByRef GapMax As Double, _
    oRebar As IRebarBody, MinParam As String, MaxParam As String)

    If ArrList.Count > 0 Then
        
        ArrList.SortItems False
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim TempxMin As Double
        
        Dim TempGap As Double

        Dim LowerIndex As Integer
        Dim UpperIndex As Integer
        
        Dim IsLowerIndexFound As Boolean
        LowerIndex = GetIndexGreaterthanthisVal(vItems, CallByName(oRebar, MinParam, VbGet), IsLowerIndexFound)
        
        Dim IsUpperIndexFound As Boolean
        UpperIndex = GetIndexLesserthanthisVal(vItems, CallByName(oRebar, MaxParam, VbGet), IsUpperIndexFound)
        
        If IsLowerIndexFound And IsUpperIndexFound Then
        
            TempGap = vItems(LowerIndex) - CallByName(oRebar, MinParam, VbGet)
            GapMin = CallByName(oRebar, MinParam, VbGet)
            GapMax = vItems(LowerIndex)
            
            Dim i As Integer
            For i = LowerIndex To UpperIndex

                If i <> UpperIndex Then
                    
                    If Abs(vItems(i + 1) - vItems(i)) > TempGap Then
                        
                        GapMin = vItems(i)
                        GapMax = vItems(i + 1)
                        TempGap = Abs(vItems(i + 1) - vItems(i))
                            
                    End If
                
                End If
                
            Next i
            
        Else
        
            GapMin = CallByName(oRebar, MinParam, VbGet)
            GapMax = CallByName(oRebar, MaxParam, VbGet)
            
        End If
        

        
    Else
    
        GapMin = CallByName(oRebar, MinParam, VbGet)
        GapMax = CallByName(oRebar, MaxParam, VbGet)
    
    End If

End Sub

Function GetIndexGreaterthanthisVal(vItems As Variant, Val As Double, ByRef IsFound As Boolean)

    Dim i As Integer
    IsFound = False
    
    For i = LBound(vItems) To UBound(vItems)
    
        If vItems(i) > Val Then
            
            GetIndexGreaterthanthisVal = i
            IsFound = True
            Exit For
            
        End If
    
    Next i
End Function

Function GetIndexLesserthanthisVal(vItems As Variant, Val As Double, ByRef IsFound As Boolean)

    Dim i As Integer
    IsFound = False
    
    For i = UBound(vItems) To LBound(vItems) Step -1
    
        If vItems(i) < Val Then
            
            GetIndexLesserthanthisVal = i
            IsFound = True
            Exit For
            
        End If
    
    Next i
End Function

Function GetDimDictDataWithKey(DimDict As Scripting.Dictionary, KeyVal As String) As IArrList

    Set GetDimDictDataWithKey = New IArrList
    
    If DimDict.Count > 0 Then
    
        If DimDict.Exists(KeyVal) Then
        
            Set GetDimDictDataWithKey = DimDict.Item(KeyVal)
        
        End If
        
    End If

End Function

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

