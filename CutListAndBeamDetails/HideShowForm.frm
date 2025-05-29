VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} HideShowForm 
   Caption         =   "Hide/ Show Components"
   ClientHeight    =   1440
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   3408
   OleObjectBlob   =   "HideShowForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "HideShowForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False











Option Explicit

Dim swSketchMgr As SldWorks.SketchManager

Private Sub CloseButton_Click()
    
    Unload Me
    
End Sub

Private Function GetOppositeVector(Dir As Variant) As Double()

    Dim Temp(2) As Double
    Dim i As Integer
    For i = LBound(Dir) To UBound(Dir)
    
        Temp(i) = -1 * Dir(i)
    
    Next i
    
    GetOppositeVector = Temp
End Function

Private Sub CreateButton_Click()

    Me.Hide

    Dim ProjectNo As String
    ProjectNo = DrawingForm.ProjectNoBox.Value
    
    Unload DrawingForm

    Set swMathUtility = swApp.GetMathUtility

    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\METAL FAB DRAWING.DRWDOT", 0, 0, 0)

    Set swSketchMgr = swDrawing.SketchManager

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)

    Dim swTopView As SldWorks.View
    Set swTopView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), "*Top", 0.21593179, 0.19172741, 0)

    Dim oFloorComp As IComp
    Set oFloorComp = New IComp
    
    oFloorComp.Initialize swFloorWeldment, swTopView
    
    Dim ViewWidth As Double
    ViewWidth = oFloorComp.xMax - oFloorComp.xMin
    
    Dim ViewHeight As Double
    ViewHeight = oFloorComp.yMax - oFloorComp.yMin
    
    Call RotateAndScaleView(swDrawing, swTopView, ViewWidth, ViewHeight)
    
    Call oFloorComp.GetBodiesList(swTopView)
    
    Dim VerticalWeldBodyList As IArrListObject
    Set VerticalWeldBodyList = oFloorComp.VerticalBodyList

    Dim HorizontalWeldBodyList As IArrListObject
    Set HorizontalWeldBodyList = oFloorComp.HorizontalBodyList
    
' Horizontal Weld Body List sorted with xMin and Vertical with yMin
    
    Call AddPerimeterBeamProperty(HorizontalWeldBodyList, "yMin", "yMax")
    Call AddPerimeterBeamProperty(VerticalWeldBodyList, "xMin", "xMax")
    
    Dim visibleVerticalMinList As IArrListObject
    Set visibleVerticalMinList = GetVisibleBodiesList(VerticalWeldBodyList.Items, "xMin", "xMax")
    
    Dim visibleHorizontalMinList As IArrListObject
    Set visibleHorizontalMinList = GetVisibleBodiesList(HorizontalWeldBodyList.Items, "yMin", "yMax")
    
    Dim visibleVerticalMaxList As IArrListObject
    Set visibleVerticalMaxList = visibleVerticalMinList.Clone
    
    visibleVerticalMaxList.SortItems "xMax", False
    
    Dim visibleHorizontalMaxList As IArrListObject
    Set visibleHorizontalMaxList = visibleHorizontalMinList.Clone
    
    visibleHorizontalMaxList.SortItems "yMax", False
    
    Dim xMinDict As Scripting.Dictionary
    Dim xMinIndexDict As Scripting.Dictionary
    Set xMinDict = GetPointDataWithIndex(visibleVerticalMinList, "xMin", xMinIndexDict)
    
    Dim xMaxDict As Scripting.Dictionary
    Dim xMaxIndexDict As Scripting.Dictionary
    Set xMaxDict = GetPointDataWithIndex(visibleVerticalMaxList, "xMax", xMaxIndexDict)
    
    Dim yMinDict As Scripting.Dictionary
    Dim yMinIndexDict As Scripting.Dictionary
    Set yMinDict = GetPointDataWithIndex(visibleHorizontalMinList, "yMin", yMinIndexDict)
    
    Dim yMaxDict As Scripting.Dictionary
    Dim yMaxIndexDict As Scripting.Dictionary
    Set yMaxDict = GetPointDataWithIndex(visibleHorizontalMaxList, "yMax", yMaxIndexDict)

    Call FindAndAddBeforeSubWeldments(xMinDict, xMinIndexDict, visibleHorizontalMinList.Clone, "xMin")
    Call FindAndAddAfterSubWeldments(xMaxDict, xMaxIndexDict, visibleHorizontalMinList.Clone, "xMax")

    Call FindAndAddBeforeSubWeldments(yMinDict, yMinIndexDict, visibleVerticalMinList.Clone, "yMin")
    Call FindAndAddAfterSubWeldments(yMaxDict, yMaxIndexDict, visibleVerticalMinList.Clone, "yMax")
    
    Call AddCallouts(visibleVerticalMinList, xMinIndexDict, xMaxIndexDict, swDrawing, swTopView)
    
    
'    Dim ConsolidatedVerticalBeamList As Scripting.Dictionary
'    Set ConsolidatedVerticalBeamList = GetConsolidatedBeamListOnly(VerticalWeldBodyList, "xMin", "xMax")
'
'    Call FindAndAddSubWeldments(ConsolidatedVerticalBeamList, HorizontalWeldBodyList.Clone, "xMin")
'
'    Dim ConsolidatedHorizontalBeamList As Scripting.Dictionary
'    Set ConsolidatedHorizontalBeamList = GetConsolidatedBeamListOnly(HorizontalWeldBodyList, "yMin", "yMax")
'
'    Call FindAndAddSubWeldments(ConsolidatedHorizontalBeamList, VerticalWeldBodyList.Clone, "yMin")
    
    
    Debug.Print ""
'

'
'    Dim VerticalSubWeldmentDict As Scripting.Dictionary
'    Set VerticalSubWeldmentDict = GetSubWeldmentDict(VerticalBeamList, HorizontalWeldBodyList)

    'Debug.Print WeldBodyList.Count

'    Dim FlatCompList As Variant
'    Dim DetailedCompList As Variant
'    Dim MaxCompHeight As Double
'    DetailedCompList = GetComponentsSortedWithXPosition(CompList.Items, FlatCompList, swFrontView, MaxCompHeight)
'
'    Dim vConsolidatedList As Variant
'
'    Dim DoorOrHVACList As IArrListObject
'    Set DoorOrHVACList = New IArrListObject
'
'    vConsolidatedList = GetConsolidatedList(DetailedCompList, DoorOrHVACList)
'
'    Set zChannelList = GetChannelCompsWithPos(zChannelList, swFrontView)
'    Set cChannelList = GetChannelCompsWithPos(cChannelList, swFrontView)
'    Set lAngleList = GetChannelCompsWithPos(lAngleList, swFrontView)
'
'    Call CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList, zChannelList, True)
'    Call CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList, cChannelList)
'    Call CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList, lAngleList, IsLAngle:=True)
'
'    swDrawing.ActivateView swFrontView.Name
'
'    Dim IsMakeUpExists As Boolean
'    Dim subAssyCompDict As Scripting.Dictionary
'    Set subAssyCompDict = AddSubAssyComponentsToDictionary(subAssyEndComponents)
'
'    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
'
'    Call AddCallouts(vConsolidatedList, swDrawing, swFrontView, MaxCompHeight, IsMakeUpExists, subAssyCompDict)
'
'    Dim Is12GAPanelExists As Boolean
'    Dim IsAllPanels12GA As Boolean
'    Is12GAPanelExists = Add12GACircles(FlatCompList, swDrawing, swBottomView, wallName, IsAllPanels12GA)
'
'    Call UpdateBottomViewPosition(FlatCompList, swDrawing, swBottomView)
'
'    Dim swLeftEdge As SldWorks.Edge
'    Dim swRightEdge As SldWorks.Edge
'
'    Dim swBottomEdge As SldWorks.Edge
'    Set swBottomEdge = AddDimensionInFrontView(swFrontView, FlatCompList, DetailedCompList, MaxHeightComp, swDrawing, swLeftEdge, swRightEdge)
'
'    Dim FlatCompDict As Scripting.Dictionary
'    Dim CompNoDict As New Scripting.Dictionary
'    Set FlatCompDict = GetCompDictionary(FlatCompList, CompNoDict)
'
'    Dim subAssylist As IArrListObject
'    Set subAssylist = New IArrListObject
'
'    If Not IsEmpty(subAssyEndComponents) Then
'
'        Dim vSubAssyComponentsIdx As Variant
'        vSubAssyComponentsIdx = GetSubAssyComponentsIndexSorted(subAssyEndComponents, CompNoDict)
'
'        Set subAssylist = AddSplitLines(vSubAssyComponentsIdx, swDrawing, swFrontView, FlatCompDict, CompNoDict, True, swLeftEdge, swRightEdge, False)
'        Call AddSplitLines(vSubAssyComponentsIdx, swDrawing, swBottomView, FlatCompDict, CompNoDict, False, swLeftEdge, swRightEdge)
'
'        Call CheckAndAddDoorOrHVACAssy(subAssylist, DoorOrHVACList, CompNoDict)
'
'
'    End If
'
'    Dim oSubAssy As ISubAssy
'    Set oSubAssy = New ISubAssy
'
'    Set oSubAssy.StartComp = FlatCompDict.Items(0)
'    Set oSubAssy.EndComp = FlatCompDict.Items(UBound(FlatCompDict.Items))
'    Set oSubAssy.StartEdge = swLeftEdge
'    Set oSubAssy.EndEdge = swRightEdge
'    Set oSubAssy.BottomEdge = swBottomEdge
'
'    oSubAssy.StartIdx = 0
'    oSubAssy.EndIdx = UBound(FlatCompDict.Items)
'    Call oSubAssy.AddDoororHVACList(DoorOrHVACList)
'
'    subAssylist.AddtoList oSubAssy
'
'    Dim Countourlist As IArrListObject
'    Set Countourlist = AddCrossMarkForAssyCuts(FlatCompDict.Items, swFrontView, swDrawing, oSubAssy)
'
'    Call AddCrossMarkForDoor(oSubAssy, swFrontView, swDrawing)
'
'    Dim UniqueHVACDict As Scripting.Dictionary
'    Set UniqueHVACDict = AddCrossMarkForHVAC(oSubAssy, swFrontView, swDrawing)
'
'    Dim NoteCount As Integer
'    Dim AssyNoteNo As Integer
'    Call AddStructuralNotes(swDrawing, swSheet, Is12GAPanelExists, IsAllPanels12GA, IsZChannelExists, NoteCount, wallName, Countourlist.Count)
'
'    Dim IsSectionViewNeeded As Boolean
'    IsSectionViewNeeded = False
'    Dim GapForSection As Double
'
'    If oSubAssy.GetWidth <= (15.75 - 2.5 * (UBound(UniqueHVACDict.Items) + 1)) * 0.0254 Then
'
'        IsSectionViewNeeded = True
'        GapForSection = (15.75 * 0.0254 - oSubAssy.GetWidth) / 2
'
'    End If
'
'    Dim MaxClearance As Double
'    Call AddDimensionsForDoororHVACInEachSubAssy(subAssylist, swDrawing, swFrontView, MaxClearance, IsSectionViewNeeded)
'    Call AddDimensionNames(subAssylist, wallName, swFrontView)
'    Call AddVerticalDimensionsForDoor(oSubAssy.GetDoorAssemblies, swFrontView, swDrawing, NoteCount)
'
'    Call AddVerticalDimensionsForHVAC(UniqueHVACDict.Items, swFrontView, swDrawing, oSubAssy, IsSectionViewNeeded, GapForSection)
'
'    Call SketchLineForNonCornerPanels(swFrontView, wallName, swDrawing, oSubAssy, NoteCount, swBottomEdge, MaxClearance)
'    Call CleanUpActivateAndAddViewLabel(swDrawing, swFrontView, wallName, oSubAssy.StartComp.yMin - MaxClearance - 0.0075, (oSubAssy.StartComp.xMin + oSubAssy.EndComp.xMax) / 2)
'
'    Call UpdateFrontViewPosition(FlatCompDict.Items, swDrawing, swFrontView)

    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
    
    Unload Me

End Sub

Function GetEdgeInViewForBody(swComp As SldWorks.Component2, oBody As ISolidBody, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    
    Dim vPointMin(2) As Double
    vPointMin(0) = oBody.xMin
    vPointMin(1) = oBody.yMin
    vPointMin(2) = oBody.zMin
    
    Dim vPointMax(2) As Double
    vPointMax(0) = oBody.xMax
    vPointMax(1) = oBody.yMax
    vPointMax(2) = oBody.zMax
    

    Call GetMaxMinPoint(vPointMin(0), vPointMax(0), xMin, xMax)
    Call GetMaxMinPoint(vPointMin(1), vPointMax(1), yMin, yMax)
    
    Dim idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        idx = 0
        
        If IsMax Then
        
            ValToMatch = xMax
            
        Else
        
             ValToMatch = xMin
             
        End If
        
    End If

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
        
    Else
    
        vEnts = oBody.GetBody.GetEdges
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)

            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
            If swCurve.IsLine Then
            
                Dim vStartPoint As Variant
                vStartPoint = swEdge.GetStartVertex.GetPoint
                vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
                
                Dim vEndPoint As Variant
                vEndPoint = swEdge.GetEndVertex.GetPoint
                vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)
                
                If Abs(vStartPoint(idx) - vEndPoint(idx)) <= 0.00001 And Abs(vStartPoint(idx) - ValToMatch) <= 0.00001 Then
                    
                    Dim vCurveParam As Variant
                    vCurveParam = swEdge.GetCurveParams2

                    If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                        
                        TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                        Set GetEdgeInViewForBody = swEdge
                        
                    End If
                    
                End If
            
            End If
            
        Next i

    End If

End Function

Private Sub GetNextAndPrevPoints()



End Sub


Private Sub AddCallouts(ArrList As IArrListObject, MinDict As Scripting.Dictionary, _
            MaxDict As Scripting.Dictionary, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
            MinParam As String, MaxParam As String, IsVertical As Boolean)

    
    Dim i As Integer
    
    Dim vItems As Variant
    vItems = ArrList.Items
    
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Dim MinIdx As Integer
        MinIdx = GetIndex(MinDict, oWeldBody, MinParam)
        
        Dim MaxIdx As Integer
        MaxIdx = GetIndex(MaxDict, oWeldBody, MaxParam)

        Dim AfterArrList As IArrListObject
        Set AfterArrList = oWeldBody.AfterSubWeldments
        
        Dim BeforeArrList As IArrListObject
        Set BeforeArrList = oWeldBody.BeforeSubWeldments
        
        If oWeldBody.IsAfterSubweldment And oWeldBody.IsBeforeSubweldment Then
        
            If AfterArrList.Count > BeforeArrList.Count Then
            
                
            
            Else
            
            
            End If
        
        ElseIf oWeldBody.IsAfterSubweldment Then
        
        
        ElseIf oWeldBody.IsBeforeSubweldment Then
        
           
        
        
        Else
            

        
            
        
        
        End If
        
        
    
    
    Next i
    
    
    
    Const Increment As Double = 0.005
    Const MaxBalloonWidth As Double = 0.015875

  
    


    
    BalloonCount = maxNoOfBalloons

    If InStr(vConsolidatedList(0).Comp.GetCustomProperty("Profile"), "CORNER") > 0 Then

        AddorSub = 1
        BalloonCount = 1

    End If

    Dim AnnXPos As Double
    Dim AnnYPos As Double
    Dim PrevXPos As Double
    Dim ChangeForNoOFTime As Boolean
    ChangeForNoOFTime = False

    Dim NoOfTimes As Integer
    NoOfTimes = 0

    Dim i As Integer
    For i = LBound(vConsolidatedList) To UBound(vConsolidatedList)

        Dim oList As IConsolidatedList
        Set oList = vConsolidatedList(i)

        Dim oComp As IComp
        Set oComp = oList.Comp

        swDrawing.ClearSelection2 True

        Dim xPos As Double
        Dim YPos As Double


        xPos = oComp.xMin + 4 * 0.0254 * swView.ScaleDecimal  '(oComp.xMin + oComp.xMax) / 2 - Abs((oComp.xMin - oComp.xMax) / 2) + 3.5 * 0.0254 * swView.ScaleDecimal
        YPos = 0.075 * oComp.yMin + 0.925 * oComp.yMax

        If oComp.IsTop Then

            If Not (i = LBound(vConsolidatedList)) Then

                Dim prevComp As IComp
                Set prevComp = vConsolidatedList(i - 1).Comp

                If AddorSub = -1 Then

                    If Abs(PrevXPos - oComp.xMin) > 2 * MaxBalloonWidth Or _
                        Abs(PrevXPos - oComp.xMin) > MaxBalloonWidth And BalloonCount > 2 Then

                        AddorSub = 1
                        BalloonCount = 1

                    ElseIf Abs(PrevXPos - oComp.xMax) > MaxBalloonWidth And BalloonCount >= 1 Then

                        AddorSub = 1
                        NoOfTimes = BalloonCount
                        BalloonCount = 1
                        ChangeForNoOFTime = True
                        xPos = oComp.xMax - 4 * 0.0254 * swView.ScaleDecimal

                    End If

                Else

                    If Abs(PrevXPos - oComp.xMin) > MaxBalloonWidth Then

                        AddorSub = 1
                        BalloonCount = 1

                    ElseIf Abs(PrevXPos - oComp.xMax) > MaxBalloonWidth And BalloonCount > 1 Then

                        BalloonCount = 1
                        xPos = oComp.xMax - 4 * 0.0254 * swView.ScaleDecimal

                    End If

                End If

                If subAssyCompDict.Exists(prevComp.GetComponent.Name2) Then

                    AddorSub = -1
                    xPos = oComp.xMin + 0.375 * Abs(oComp.xMin - oComp.xMax)

                    If Not (i = UBound(vConsolidatedList)) Then

                        Dim NextComp As IComp
                        Set NextComp = vConsolidatedList(i + 1).Comp

                        If (Abs(NextComp.xMin - oComp.xMin) > MaxBalloonWidth) Then

                           BalloonCount = 1

                        End If

                    End If

                End If

            End If

            If ChangeForNoOFTime Then

                NoOfTimes = NoOfTimes - 1

                If NoOfTimes < 0 Then

                    ChangeForNoOFTime = False
                    BalloonCount = maxNoOfBalloons
                    AddorSub = -1

                End If

            End If

            AnnXPos = xPos

            If AddorSub = 1 Then

                If BalloonCount > maxNoOfBalloons Then

                    AddorSub = -1
                    BalloonCount = BalloonCount + AddorSub



                End If

            Else

                If BalloonCount < 1 Then

                    xPos = oComp.xMax - 4 * 0.0254 * swView.ScaleDecimal '(oComp.xMin + oComp.xMax) / 2 + Abs((oComp.xMin - oComp.xMax) / 2) - 3.5 * 0.0254 * swView.ScaleDecimal
                    AnnXPos = xPos

                    If oList.Qty > 2 Then

                        BalloonCount = 1
                        AnnXPos = xPos + 0.5 * (oComp.xMax - oComp.xMin)

                    Else

                        BalloonCount = maxNoOfBalloons

                    End If

                End If

            End If

            AnnYPos = MaxCompHeight + BalloonCount * Increment
            BalloonCount = BalloonCount + AddorSub
            PrevXPos = xPos



        ElseIf oComp.IsBottom Then

            xPos = (oComp.xMin + oComp.xMax) / 2
            YPos = 0.7 * oComp.yMin + 0.3 * oComp.yMax
            AnnXPos = xPos
            AnnYPos = oComp.yMin - Increment

        Else

            xPos = (oComp.xMin + oComp.xMax) / 2
            YPos = 0.3 * oComp.yMin + 0.7 * oComp.yMax
            AnnXPos = oComp.xMin - 3 * Increment
            AnnYPos = YPos - 2 * Increment

        End If


        Dim IsSelected As Boolean
        IsSelected = False
        Call SelectComponent(swDrawing, oComp, xPos, YPos, 1, IsSelected, swView)

        If IsSelected Then

            Dim swComp As SldWorks.Component2
            Set swComp = oComp.GetComponent
            'Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))

            Dim swAnn As SldWorks.Annotation
            Set swAnn = InsertBalloonAndGetAnnotations(swDrawing, oList.Qty, AnnXPos, AnnYPos)

            If Not swAnn Is Nothing Then

                Dim swNote As SldWorks.Note
                Set swNote = swAnn.GetSpecificAnnotation

                Dim HeadStyle As Integer

                swAnn.SetLeader3 swLeaderStyle_e.swAlwaysAttachToBalloon + swLeaderStyle_e.swSTRAIGHT, swLeaderSide_e.swLS_SMART, False, False, True, False
                HeadStyle = swAnn.SetArrowHeadStyleAtIndex(0, swArrowStyle_e.swCLOSED_ARROWHEAD)

                'Debug.Print HeadStyle

                If oComp.IsTop Then

                    If AddorSub = 1 Then

                        Dim vNoteExtents As Variant
                        vNoteExtents = swNote.GetExtent

                        If oList.Qty > 1 Then

                             AnnXPos = xPos - ((vNoteExtents(3) - vNoteExtents(0))) + 0.0064

                        Else
                            AnnXPos = xPos - ((vNoteExtents(3) - vNoteExtents(0))) + 0.0027

                        End If

                        swAnn.SetPosition2 AnnXPos, AnnYPos, 0

                    End If

                End If

            End If

        End If

        Call AddHatchForMakeUpPanel(oComp, swDrawing, IsMakeUpExists, swView)

    Next i

    Call UpdateHatchProperties(swView)

End Sub

Function GetIndex(Dict As Scripting.Dictionary, oWeldBody As IWeldBody, Param As String)
    
    Dim keyVal As Double
    keyVal = CallByName(oWeldBody, Param, VbGet)
    
    If Dict.Exists(keyVal) Then
    
        GetIndex = Dict.Item(keyVal)
        
    Else
    
        Dim vKeys As Variant
        vKeys = Dict.Keys
        
        For i = LBound(vKeys) To UBound(vKeys)

            If Abs(vKeys(i) - keyVal) <= 0.0001 Then
            
                GetIndex = i
                Exit For
                
            End If
            
        Next i

End Function

Function GetPointDataWithIndex(ArrList As IArrListObject, Parameter As String, ByRef IndexDict As Scripting.Dictionary) As Scripting.Dictionary

    Set GetPointDataWithIndex = New Scripting.Dictionary
    Set IndexDict = New Scripting.Dictionary
    
    Dim i As Integer
    Dim vItems As Variant
    vItems = ArrList.Items
    
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Dim keyVal As Double
        keyVal = CallByName(oWeldBody, Parameter, VbGet)
        
        If GetPointDataWithIndex.Exists(keyVal) Then
            
             GetPointDataWithIndex.Item(keyVal).AddtoList oWeldBody
        
        Else
            
            Dim NewArrList As IArrListObject
            Set NewArrList = New IArrListObject
            
            NewArrList.AddtoList oWeldBody
            
            If GetPointDataWithIndex.Count = 0 Then

                GetPointDataWithIndex.Add keyVal, NewArrList
                IndexDict.Add keyVal, IndexDict.Count
                
            Else
            
                Dim PrevKey As Double
                PrevKey = GetPointDataWithIndex.Keys(GetPointDataWithIndex.Count - 1)
                
                If Abs(PrevKey - keyVal) <= 0.0001 Then
                    
                    GetPointDataWithIndex.Item(PrevKey).AddtoList oWeldBody
                    
                Else
                
                    GetPointDataWithIndex.Add keyVal, NewArrList
                    IndexDict.Add keyVal, IndexDict.Count
                
                End If

            End If
            
        End If

    Next i
    
End Function

Function GetVisibleBodiesList(vItems As Variant, ParamMin As String, ParamMax As String) As IArrListObject

    Set GetVisibleBodiesList = New IArrListObject
    
    Dim i As Integer
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Debug.Print oWeldBody.GetBody.Name
        Debug.Print oWeldBody.Cutlist.Description

        If i = 0 Then
        
            GetVisibleBodiesList.AddtoList oWeldBody
        
        Else
        
            If False = CheckWhetherTheBodyisWithInAnotherBody(oWeldBody, GetVisibleBodiesList.Items, GetVisibleBodiesList.Count - 1, ParamMin, ParamMax) Then
            
                GetVisibleBodiesList.AddtoList oWeldBody
            
            End If
        
        End If
    
    Next i
    
End Function

Private Function CheckWhetherTheBodyisWithInAnotherBody(WeldBodyToCheck As IWeldBody, vItems As Variant, idx As Integer, keyParamMin, keyParamMax)

    CheckWhetherTheBodyisWithInAnotherBody = False

    Dim i As Integer
    For i = idx To LBound(vItems) Step -1
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        If Not Left(oWeldBody.Cutlist.Description, 1) = "L" Then

            If CallByName(oWeldBody, keyParamMax, VbGet) < CallByName(WeldBodyToCheck, keyParamMin, VbGet) Then
            
                Exit For
                
            End If
            
            If (WeldBodyToCheck.xMin > oWeldBody.xMin Or Abs(WeldBodyToCheck.xMin - oWeldBody.xMin) <= 0.0001) And _
                (WeldBodyToCheck.xMax < oWeldBody.xMax Or Abs(WeldBodyToCheck.xMax - oWeldBody.xMax) <= 0.0001) And _
                (WeldBodyToCheck.yMin > oWeldBody.yMin Or Abs(WeldBodyToCheck.yMin - oWeldBody.yMin) <= 0.0001) And _
                (WeldBodyToCheck.yMax < oWeldBody.yMax Or Abs(WeldBodyToCheck.yMax - oWeldBody.yMax) <= 0.0001) And _
                (WeldBodyToCheck.zMin > oWeldBody.zMin Or Abs(WeldBodyToCheck.zMin - oWeldBody.zMin) <= 0.0001) And _
                (WeldBodyToCheck.zMax < oWeldBody.zMax Or Abs(WeldBodyToCheck.zMax - oWeldBody.zMax) <= 0.0001) Then
                
                CheckWhetherTheBodyisWithInAnotherBody = True
                Exit For
                
            End If
            
        End If

    Next i

End Function

Sub AddPerimeterBeamProperty(ArrList As IArrListObject, SortParamMin As String, SortParamMax As String) 'Sort with Min Parameter

    ArrList.SortItems SortParamMax, True
    ArrList.Items(0).IsPerimeter = True
    

    ArrList.SortItems SortParamMin, False
    ArrList.Items(0).IsPerimeter = True

End Sub

Sub FindAndAddBeforeSubWeldments(Dict As Scripting.Dictionary, DictIndex As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String)
    
    Dim CheckParameterMin As String
    Dim CheckParameterMax As String
    
    If Left(Parameter, 1) = "x" Then
    
        CheckParameterMin = "yMin"
        CheckParameterMax = "yMax"
        
    Else
        
        CheckParameterMin = "xMin"
        CheckParameterMax = "xMax"
        
    End If
        
    If ArrList.Count > 0 Then
    
        ArrList.SortItems Parameter, False
        
        Dim i As Integer
        Dim vItems As Variant
        vItems = ArrList.Items

        For i = LBound(vItems) To UBound(vItems)
            
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            Dim Index As Integer
            Index = GetIndexWhenKeyValExceedThisValue(DictIndex, CallByName(oWeldBody, Parameter, VbGet))
            
            Dim FirstWeldBody As IWeldBody
            Set FirstWeldBody = GetWeldBodyAttachedBeforeThisBody(oWeldBody, Dict, Index - 1, CheckParameterMin, CheckParameterMax)
            
            'Debug.Print oWeldBody.GetBody.Name
            'Debug.Print FirstWeldBody.GetBody.Name
            
            If Not FirstWeldBody Is Nothing Then
            
                Call FirstWeldBody.AddToSubWeldmentList(True, oWeldBody)
                
            End If
            
        Next i

    End If

End Sub

Sub FindAndAddAfterSubWeldments(Dict As Scripting.Dictionary, DictIndex As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String)
    
    Dim CheckParameterMin As String
    Dim CheckParameterMax As String
    
    If Left(Parameter, 1) = "x" Then
    
        CheckParameterMin = "yMin"
        CheckParameterMax = "yMax"
        
    Else
        
        CheckParameterMin = "xMin"
        CheckParameterMax = "xMax"
        
    End If
    
    If ArrList.Count > 0 Then
    
        ArrList.SortItems Parameter, False
        
        Dim i As Integer
        Dim vItems As Variant
        vItems = ArrList.Items

        For i = LBound(vItems) To UBound(vItems)
            
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            Dim Index As Integer
            Index = GetIndexWhenKeyValExceedThisValue(DictIndex, CallByName(oWeldBody, Parameter, VbGet))
            
            Dim SecondWeldBody As IWeldBody
            Set SecondWeldBody = GetWeldBodyAttachedAfterThisBody(oWeldBody, Dict, Index, CheckParameterMin, CheckParameterMax)
            
            If Not SecondWeldBody Is Nothing Then
            
                Call SecondWeldBody.AddToSubWeldmentList(False, oWeldBody)
                
            End If
            'Debug.Print oWeldBody.GetBody.Name
            'Debug.Print SecondWeldBody.GetBody.Name
            
        Next i

    End If

End Sub

Function GetIndexWhenKeyValExceedThisValue(Dict As Scripting.Dictionary, ValToCheck As Double)


    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        If vKeys(i) > ValToCheck Then
        
            GetIndexWhenKeyValExceedThisValue = i
            Exit For
            
        End If

    Next i

End Function


Function GetWeldBodyAttachedBeforeThisBody(WeldBodyToCheck As IWeldBody, Dict As Scripting.Dictionary, _
    idx As Integer, CheckParameterMin As String, CheckParameterMax As String) As IWeldBody

    Dim i As Integer
    Dim j As Integer
    
    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    For j = idx To LBound(vKeys) Step -1
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(j))
    
        Dim vItems As Variant
        vItems = ArrList.Items
    
        For i = LBound(vItems) To UBound(vItems)
    
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            If (CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) > CallByName(oWeldBody, CheckParameterMin, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) - CallByName(oWeldBody, CheckParameterMin, VbGet)) <= 0.0001) And _
                (CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) < CallByName(oWeldBody, CheckParameterMax, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) - CallByName(oWeldBody, CheckParameterMax, VbGet)) <= 0.0001) Then
        
                If (oWeldBody.zMin < WeldBodyToCheck.zMin Or Abs(oWeldBody.zMin - WeldBodyToCheck.zMin) <= 0.0001) And oWeldBody.zMax > WeldBodyToCheck.zMin Then
        
                    Set GetWeldBodyAttachedBeforeThisBody = oWeldBody
                    Exit Function
                
                End If
    
            End If
        
         Next i
         
    Next j

End Function


Function GetWeldBodyAttachedAfterThisBody(WeldBodyToCheck As IWeldBody, Dict As Scripting.Dictionary, _
    idx As Integer, CheckParameterMin As String, CheckParameterMax As String) As IWeldBody

    Dim i As Integer
    Dim j As Integer
    
    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    For j = idx To UBound(vKeys)
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(j))
    
        Dim vItems As Variant
        vItems = ArrList.Items
    
        For i = LBound(vItems) To UBound(vItems)
    
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            If (CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) > CallByName(oWeldBody, CheckParameterMin, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) - CallByName(oWeldBody, CheckParameterMin, VbGet)) <= 0.0001) And _
                (CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) < CallByName(oWeldBody, CheckParameterMax, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) - CallByName(oWeldBody, CheckParameterMax, VbGet)) <= 0.0001) Then
        
                If (oWeldBody.zMin < WeldBodyToCheck.zMin Or Abs(oWeldBody.zMin - WeldBodyToCheck.zMin) <= 0.0001) And oWeldBody.zMax > WeldBodyToCheck.zMin Then
        
                    Set GetWeldBodyAttachedAfterThisBody = oWeldBody
                    Exit Function
                
                End If
    
            End If
        
         Next i
         
    Next j

End Function


'Function SelectEdgeWithSelectData(swEdge As SldWorks.Edge, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
'                swComp As SldWorks.Component2, ByRef SelXPos As Double, ByRef SelYPos As Double, Optional PercentageFromStart As Double = 0.5) As Boolean
'
'    Dim swSelectMgr As SldWorks.SelectionMgr
'    Set swSelectMgr = swDrawing.SelectionManager
'
'    Dim swSelectData As SldWorks.SelectData
'    Set swSelectData = swSelectMgr.CreateSelectData
'
'    Dim vStartPoint As Variant
'    vStartPoint = swEdge.GetStartVertex.GetPoint
'    vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
'
'    Dim vEndPoint As Variant
'    vEndPoint = swEdge.GetEndVertex.GetPoint
'    vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)
'
'    Dim swMathStartPoint As SldWorks.MathPoint
'    Set swMathStartPoint = swMathUtility.CreatePoint(vStartPoint)
'
'    Dim swMathEndPoint As SldWorks.MathPoint
'    Set swMathEndPoint = swMathUtility.CreatePoint(vEndPoint)
'
'    Dim swPosVector As SldWorks.MathVector
'    Set swPosVector = swMathEndPoint.Subtract(swMathStartPoint)
'
'    Set swMathStartPoint = swMathStartPoint.AddVector(swPosVector.Scale(PercentageFromStart))
'
'    SelXPos = swMathStartPoint.ArrayData(0)
'    SelYPos = swMathStartPoint.ArrayData(1)
'
'    swSelectData.View = swView
'    swSelectData.X = SelXPos '(vStartPoint(0) + vEndPoint(0)) / 2
'    swSelectData.Y = SelYPos 'vStartPoint(1)
'
'    Dim swEntity As SldWorks.Entity
'    Set swEntity = swEdge
'
'    SelectEdgeWithSelectData = swEntity.Select4(False, swSelectData)
'
'End Function
'
'Sub AddQtyToDimension(swDisplayDim As SldWorks.DisplayDimension, Qty As Integer)
'
'    If Qty > 1 Then
'
'        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X "
'
'    End If
'
'End Sub
'
'Sub AddCrossMarkForDoor(oSubAssy As ISubAssy, swView As SldWorks.View, _
'                swDrawing As SldWorks.DrawingDoc)
'
'    Dim vDoorAssy As Variant
'    vDoorAssy = oSubAssy.GetDoorAssemblies
'
'    If Not IsEmpty(vDoorAssy) Then
'
'        swDrawing.ActivateSheet swDrawing.GetCurrentSheet.GetName
'        swDrawing.ActivateView swView.Name
'
'        swView.FocusLocked = True
'
'        Dim i As Integer
'        For i = LBound(vDoorAssy) To UBound(vDoorAssy)
'
'            Dim oDoorAssy As IDoorOrHVACAssy
'            Set oDoorAssy = vDoorAssy(i)
'
'            If oDoorAssy.cChannelCompList.Count = 1 Then
'
'                Dim DoorLeftEdge As SldWorks.Edge
'                Set DoorLeftEdge = GetEdgeInView(oDoorAssy.StartComp, swView, False, True)
'
'                Dim DoorRightEdge As SldWorks.Edge
'                Set DoorRightEdge = GetEdgeInView(oDoorAssy.EndComp, swView, False, False)
'
'                Dim DoorBottomEdge As SldWorks.Edge
'                Set DoorBottomEdge = GetEdgeInView(oDoorAssy.StartComp, swView, True, False)
'
'                Dim cChannelComp As IComp
'                Set cChannelComp = oDoorAssy.cChannelCompList.Items(0)
'
'                Dim DoorTopEdge As SldWorks.Edge
'                Set DoorTopEdge = GetEdgeInView(cChannelComp, swView, True, False)
'
'                Dim LowerLeftPoint(2) As Double
'                LowerLeftPoint(0) = oDoorAssy.StartComp.xMax
'                LowerLeftPoint(1) = oDoorAssy.StartComp.yMin
'                LowerLeftPoint(2) = 0
'
'                Dim vLowerLeftPoint As Variant
'                vLowerLeftPoint = GetSheetPointInViewSpace(swView, LowerLeftPoint)
'
'                Dim LowerRightPoint(2) As Double
'                LowerRightPoint(0) = oDoorAssy.EndComp.xMin
'                LowerRightPoint(1) = oDoorAssy.StartComp.yMin
'                LowerRightPoint(2) = 0
'
'                Dim vLowerRightPoint As Variant
'                vLowerRightPoint = GetSheetPointInViewSpace(swView, LowerRightPoint)
'
'                Dim UpperLeftPoint(2) As Double
'                UpperLeftPoint(0) = oDoorAssy.StartComp.xMax
'                UpperLeftPoint(1) = cChannelComp.yMin
'                UpperLeftPoint(2) = 0
'
'                Dim vUpperLeftPoint As Variant
'                vUpperLeftPoint = GetSheetPointInViewSpace(swView, UpperLeftPoint)
'
'                Dim UpperRightPoint(2) As Double
'                UpperRightPoint(0) = oDoorAssy.EndComp.xMin
'                UpperRightPoint(1) = cChannelComp.yMin
'                UpperRightPoint(2) = 0
'
'                Dim vUpperRightPoint As Variant
'                vUpperRightPoint = GetSheetPointInViewSpace(swView, UpperRightPoint)
'
'                Dim swSketchManager As SldWorks.SketchManager
'                Set swSketchManager = swDrawing.SketchManager
'
'                Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerLeftPoint, vUpperRightPoint, DoorLeftEdge, DoorRightEdge, DoorBottomEdge, DoorTopEdge)
'                Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerRightPoint, vUpperLeftPoint, DoorRightEdge, DoorLeftEdge, DoorBottomEdge, DoorTopEdge)
'
'            End If
'
'        Next i
'
'    End If
'
'End Sub
'

'
'Sub CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, FirstPoint As Variant, SecondPoint As Variant, FirstPtVerticalEdge As SldWorks.Edge, _
'                SecondPtVerticalEdge As SldWorks.Edge, FirstPtHorEdge As SldWorks.Edge, SecondPtHorEdge As SldWorks.Edge)
'
'
'
'    Dim swSketchSegment As SketchSegment
'    Set swSketchSegment = swSketchManager.CreateLine(FirstPoint(0), FirstPoint(1), FirstPoint(2), _
'                        SecondPoint(0), SecondPoint(1), SecondPoint(2))
'    swSketchSegment.ConstructionGeometry = True
'
'    If Not swSketchSegment Is Nothing Then
'
'        Dim swSketchLine As SldWorks.sketchLine
'        Set swSketchLine = swSketchSegment
'
'        Dim swFirstPoint As SldWorks.sketchPoint
'        Set swFirstPoint = swSketchLine.GetStartPoint2
'
'        Call AddCoincidentRelationbwPointAndEdge(FirstPtVerticalEdge, swFirstPoint, swDrawing, swView)
'        Call AddCoincidentRelationbwPointAndEdge(FirstPtHorEdge, swFirstPoint, swDrawing, swView)
'
'
'        Dim swSecondPoint As SldWorks.sketchPoint
'        Set swSecondPoint = swSketchLine.GetEndPoint2
'
'        Call AddCoincidentRelationbwPointAndEdge(SecondPtVerticalEdge, swSecondPoint, swDrawing, swView)
'        Call AddCoincidentRelationbwPointAndEdge(SecondPtHorEdge, swSecondPoint, swDrawing, swView)
'
'    End If
'
'End Sub
'
'Sub AddCoincidentRelationbwPointAndEdge(swEdge As SldWorks.Edge, swSketchPoint As SldWorks.sketchPoint, _
'        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
'
'    swView.SelectEntity swEdge, False
'    swSketchPoint.Select4 True, Nothing
'
'    swDrawing.SketchAddConstraints "sgCOINCIDENT"
'
'
'End Sub

'

'
'Private Sub UpdateFrontViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
'
'    Dim oStartComp As IComp
'    Set oStartComp = vComps(0)
'
'    Dim oEndComp As IComp
'    Set oEndComp = vComps(UBound(vComps))
'
'    Dim CenterX As Double
'    CenterX = (oStartComp.xMin + oEndComp.xMax) / 2
'
'    Dim viewPosition As Variant
'    viewPosition = swView.Position
'
'    viewPosition(0) = viewPosition(0) + (viewPosition(0) - CenterX)
'
'    swView.Position = viewPosition
'
'End Sub
'

'
'Sub CheckandAddLayer(LayName As String, LayerDesc As String, swLayerMgr As SldWorks.LayerMgr)
'
'    Dim vLayNames As Variant
'    vLayNames = swLayerMgr.GetLayerList
'
'    Dim IsLayerExists As Boolean
'
'    Dim i As Integer
'    For i = 0 To UBound(vLayNames)
'
'        If vLayNames(i) = LayName Then
'
'            IsLayerExists = True
'            Exit For
'
'        End If
'
'    Next i
'
'    If Not (IsLayerExists) Then
'
'        swLayerMgr.AddLayer LayName, LayerDesc, 0, swLineStyles_e.swLineDEFAULT, swLineWeights_e.swLW_NONE
'
'        Dim swLayer As SldWorks.Layer
'        Set swLayer = swLayerMgr.GetLayer(LayName)
'
'        swLayer.Style = swLineStyles_e.swLineCENTER
'        swLayer.Width = swLineWeights_e.swLW_THICK5
'
'    End If
'
'End Sub
'

'Private Sub AddOverallDimension(oSubAssy As ISubAssy, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, Clearance As Double)
'
'    Dim swDisplayDim As SldWorks.DisplayDimension
'    Set swDisplayDim = SelectAndAddDimension(oSubAssy.StartEdge, oSubAssy.EndEdge, swDrawing, _
'                oSubAssy.EndComp.xMin - 0.01, oSubAssy.EndComp.yMin - Clearance, swView)
'    Set oSubAssy.Dimension = swDisplayDim
'
'End Sub
'

'
'Private Function GetControlSketch() As SldWorks.Component2
'
'    Dim swTopLevelAssy As SldWorks.AssemblyDoc
'    Set swTopLevelAssy = swTopLevelModel
'
'    Dim vComps As Variant
'    vComps = swTopLevelAssy.GetComponents(True)
'
'    Dim i As Integer
'    For i = LBound(vComps) To UBound(vComps)
'
'        Dim swComp As SldWorks.Component2
'        Set swComp = vComps(i)
'
'        If InStr(swComp.Name2, "CONTROL") > 0 And InStr(swComp.Name2, "SKETCH") > 0 Then
'
'            Dim vBodies As Variant
'            Dim vBodiesInfo As Variant
'            vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodiesInfo)
'
'            If IsEmpty(vBodies) Then
'
'                Set GetControlSketch = swComp
'                Exit Function
'
'            End If
'
'
'        End If
'
'    Next i
'
'End Function
'
'Private Sub AddSplitLineNote(swSketchSegment As SldWorks.sketchLine, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
'            NoteText As String, Optional IsRight As Boolean = True, Optional ClearanceVal As Double = 0.005)
'
'
'    Dim vPointInSheet As Variant
'
''    If InStr(NoteText, "SPLIT") > 0 Then
'
''        vPointInSheet = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, False, 0)
''
''    Else
''
'        vPointInSheet = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, False)
'
''    End If
'
'    If IsRight Then
'
'        Call AddNoteToView(swDrawing, NoteText, vPointInSheet(0) + ClearanceVal, vPointInSheet(1) + 0.00625)
'
'    Else
'
'        Call AddNoteToView(swDrawing, NoteText, vPointInSheet(0) - ClearanceVal, vPointInSheet(1) + 0.00625)
'
'    End If
'
'End Sub
'

'
'Function GetSelectedComponents() As Variant
'
'    Dim swSelectionMgr As SldWorks.SelectionMgr
'    Set swSelectionMgr = swTopLevelModel.SelectionManager
'
'    Dim compDict As Scripting.Dictionary
'    Set compDict = New Scripting.Dictionary
'
'    If swSelectionMgr.GetSelectedObjectCount2(-1) > 0 Then
'
'        Dim i As Integer
'        For i = 0 To swSelectionMgr.GetSelectedObjectCount2(-1) - 1
'
'            Dim swComp As SldWorks.Component2
'            Set swComp = swSelectionMgr.GetSelectedObjectsComponent4(i + 1, -1)
'
'            If False = compDict.Exists(swComp.Name2) Then
'
'                compDict.Add swComp.Name2, swComp
'
'            End If
'
'        Next i
'
'    End If
'
'    If Not (compDict.Count = 0) Then
'
'        GetSelectedComponents = compDict.Items
'
'    End If
'
'End Function

'Private Sub ActivateDrawingDocument(swModel As SldWorks.ModelDoc2)
'
'    Dim swFrame As SldWorks.Frame
'    Set swFrame = swApp.Frame
'
'    Dim vModelWindows As Variant
'    vModelWindows = swFrame.ModelWindows
'
'    Dim i As Integer
'    For i = LBound(vModelWindows) To UBound(vModelWindows)
'
'        Dim swModelWindow As SldWorks.ModelWindow
'        Set swModelWindow = vModelWindows(i)
'
'        If swModelWindow.Title = swModel.GetTitle Then
'
'            swModelWindow.Activate
'            Exit Sub
'
'        End If
'
'    Next i
'End Sub


'Private Function SelectAndAddDimension(swEdge1 As SldWorks.Edge, swEdge2 As SldWorks.Edge, swDrawing As SldWorks.ModelDoc2, _
'            xPos As Double, YPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True) As SldWorks.DisplayDimension
'
'    If Not (swEdge1 Is Nothing) And Not (swEdge2 Is Nothing) Then
'
'        swDrawing.ClearSelection2 True
'        Call SelectEntity(swEdge1, False, swView)
'        Call SelectEntity(swEdge2, True, swView)
'
'        Set SelectAndAddDimension = swDrawing.AddHorizontalDimension2(xPos, YPos, 0)
'
'        If Not SelectAndAddDimension Is Nothing Then
'
'            SelectAndAddDimension.CenterText = True
'
'            If IsDual Then
'
'                SelectAndAddDimension.SetDual2 False, False
'
'            End If
'
'        End If
'
'    End If
'
'End Function
'

'Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, Is12GAPanelExists As Boolean, _
'            IsAllPanels12GA As Boolean, IsDoorExists As Boolean, ByRef NoteCount As Integer, _
'                wallName As String, AssyCutCount As Integer) As SldWorks.Note
'
'    swDrawing.ActivateSheet swSheet.GetName
'
'    Dim swStructuralNote As SldWorks.Note
'    Dim Note As String
'
'    If Is12GAPanelExists Then
'
'        NoteCount = 2
'        If IsAllPanels12GA Then
'
'            Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
'                "<FONT size=8PTS style=R>1. ALL PANELS ARE 12GA." & vbCrLf & _
'             "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
'
'        Else
'            Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
'                "<FONT size=8PTS style=R>1. ALL CIRCLED PANELS ARE 12GA." & vbCrLf & _
'             "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
'
'        End If
'
'    Else
'
'        NoteCount = 1
'        Note = "<FONT size=10PTS style=B> NOTES:" & vbCrLf & _
'            "<FONT size=8PTS style=R>1. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
'
'     End If
'
'
'    If AssyCutCount > 0 Then
'
'        NoteCount = NoteCount + 1
'        Note = Note & vbCrLf & NoteCount & ". VERIFY THE POSITION OF OEM BLOCKOUT WITH RESPECT TO L-TABS IN LINER PANEL."
'
'    End If
'
'
'    If InStr(wallName, "Wall") > 0 Then
'
'        If IsDoorExists Then
'
'            NoteCount = NoteCount + 1
'            Note = Note & vbCrLf & NoteCount & ". DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM HORIZONTAL FACE OF DOOR C-CHANNEL."
'
'        End If
'
'        NoteCount = NoteCount + 1
'        Note = Note & vbCrLf & NoteCount & ". DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM OF CEILING PANELS, USE FOR CEILING L-ANGLE PLACEMENT."
'
'    End If
'
'    Set swStructuralNote = swDrawing.CreateText2(Note, 1.99241243641486E-02, 6.92464210842187E-02, 0, 0, 0)
'    swStructuralNote.SetTextJustification swTextJustification_e.swTextJustificationLeft
'
'End Function
'
Private Sub InsertSketchBlock(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, ProjectNo As String)

    swDrawing.ActivateSheet swSheet.GetName

    Dim vSheetProp As Variant
    vSheetProp = swSheet.GetProperties

    Dim vPt(2) As Double
    vPt(0) = 0.01590679 * vSheetProp(3)
    vPt(1) = 0.00995866 * vSheetProp(3)
    vPt(2) = 0

    Dim SketchBlockInsertionPt As SldWorks.MathPoint
    Set SketchBlockInsertionPt = swMathUtility.CreatePoint(vPt)

    Dim swBlockDefinition As SldWorks.SketchBlockDefinition
    Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(SketchBlockInsertionPt, _
                "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " EXTERNAL ELEVATION KEY.SLDBLK", True, 1, 0)


End Sub

'Private Sub CleanUpActivateAndAddViewLabel(swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, wallName As String, YPos As Double, _
'    Optional xPos As Double = 0)
'
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayOrigins, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayPlanes, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints2, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCurves, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayAllAnnotations, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCompAnnotations, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLiveSections, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLights, False
'
'    swDrawing.ActivateView swView.Name
'
'    Dim SheetDesc As String
'    Dim LabelText As String
'    If InStr(wallName, "Wall") > 0 Then
'
'        SheetDesc = "STRUCTURAL, ELEVATION, EXTERNAL PANELS, " & UCase(wallName)
'        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34) & _
'         vbCrLf & "<FONT size=8PTS style=R> (INTERIOR VIEW)"
'
'    Else
'
'        SheetDesc = "STRUCTURAL, " & UCase(wallName)
'        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34)
'
'    End If
'    swDrawing.Extension.CustomPropertyManager("").Set2 "SHEET DESCRIPTION", SheetDesc
'    swDrawing.Extension.CustomPropertyManager("").Set2 "ISSUED FOR", "CONSTRUCTION"
'
'    If xPos = 0 Then
'
'        Dim vOutline As Variant
'        vOutline = swView.GetOutline
'        xPos = (vOutline(0) + vOutline(2)) / 2
'
'    End If
'
'    Dim swLabelNote As SldWorks.Note
'
'    Set swLabelNote = swDrawing.CreateText2(LabelText, xPos, YPos, 0, 0, 0)
'    swLabelNote.SetTextJustification swTextJustification_e.swTextJustificationCenter
'
'    swDrawing.Extension.Rebuild swRebuildOptions_e.swCurrentSheetDisp
'
'End Sub
'



'Sub AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, YPos As Double)
'
'    Dim swNote As SldWorks.Note
'    Set swNote = swDrawing.InsertNote(NoteText)
'
'    If Not swNote Is Nothing Then
'
'        Dim swAnnotation As SldWorks.Annotation
'        Set swAnnotation = swNote.GetAnnotation()
'
'        If Not swAnnotation Is Nothing Then
'
'            swAnnotation.SetPosition xPos, YPos, 0
'
'        End If
'
'    End If
'
'End Sub
'
'Sub CreateRibSketches(ByRef swSketchSegmentHor As SldWorks.SketchSegment, ByRef swSketchSegmentVer As SldWorks.SketchSegment, _
'                    xMin As Double, xMax As Double, yMin As Double, yMax As Double, CompPos As Integer, swSketchMgr As SldWorks.SketchManager, _
'                        OffSetVer As Double, OffSetHor As Double)
'
'    Const Length As Double = 3
'    Const FrontOffset As Double = 1.5
'
'    If CompPos = 0 Then
'
'        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMax - OffSetHor * 0.0254, yMin + FrontOffset * 0.0254, _
'                                0, xMax - (OffSetHor - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)
'
'        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMin + FrontOffset * 0.0254, yMax - OffSetVer * 0.0254, _
'                                0, xMin + FrontOffset * 0.0254, yMax - (OffSetVer - Length) * 0.0254, 0)
'
'    Else
'
'        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMin + OffSetVer * 0.0254, yMin + FrontOffset * 0.0254, _
'                                0, xMin + (OffSetVer - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)
'
'        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMax - FrontOffset * 0.0254, yMax - OffSetHor * 0.0254, _
'                                0, xMax - FrontOffset * 0.0254, yMax - (OffSetHor - Length) * 0.0254, 0)
'
'    End If
'
'End Sub
'


Function RotateAndScaleView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            ViewWidth As Double, ViewHeight As Double) As SldWorks.View
    

    If ViewHeight > ViewWidth Then
        
        swView.Angle = 1.57079632679
        
        Dim TempVal As Double
        TempVal = ViewHeight
        ViewHeight = ViewWidth
        ViewWidth = TempVal
        
    End If

    Dim xScale As Integer
    Dim yScale As Integer
    xScale = GetScaleValue(ViewWidth / (swView.ScaleDecimal * 0.371))
    yScale = GetScaleValue(ViewHeight / (swView.ScaleDecimal * 0.1295)) '0.20995
    
    Dim IsScaleSet As Boolean
    IsScaleSet = False
    
    If xScale > 0 And yScale > 0 Then
        
        If yScale > xScale Then
            
            IsScaleSet = swView.Sheet.SetScale(1, yScale, True, True)
           
        Else
            
            IsScaleSet = swView.Sheet.SetScale(1, xScale, True, True)
        
        End If
        
    End If

End Function

Function GetScaleValue(scaleVal As Double) As Integer

    GetScaleValue = 0
    
    Dim stdScales As Variant
    stdScales = Array(1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)
    
    Dim i As Integer
    For i = LBound(stdScales) To UBound(stdScales)
    
        If stdScales(i) >= scaleVal Then
           GetScaleValue = stdScales(i)
           Exit For
        End If
    
    Next i

End Function





