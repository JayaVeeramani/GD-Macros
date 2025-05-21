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
Dim xDirectionVector(2) As Double
Dim yDirectionVector(2) As Double
Dim zDirectionVector(2) As Double
Const SheetPosForLastBalloon As Double = 0.266


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

Function SelectEntity(swEnt As Object, Append As Boolean, swView As SldWorks.View) As Boolean 'swView As SldWorks.View

    Dim swEntity As SldWorks.Entity
    Set swEntity = swEnt
    
    'SelectEntity = swEntity.Select4(Append, Nothing)
    
    SelectEntity = swView.SelectEntity(swEnt, Append)
    
End Function

Private Sub CreateButton_Click()

    xDirectionVector(0) = 1
    xDirectionVector(1) = 0
    xDirectionVector(2) = 0
    
    yDirectionVector(0) = 0
    yDirectionVector(1) = 1
    yDirectionVector(2) = 0
    
    zDirectionVector(0) = 0
    zDirectionVector(1) = 0
    zDirectionVector(2) = 1

    Me.Hide
    
    Dim wallName As String
    wallName = WallDrawingForm.WallNameComboBox.Value
    
    Dim ProjectNo As String
    ProjectNo = WallDrawingForm.ProjectNoBox.Value
    
    Unload WallDrawingForm
    
    Dim viewName As String
    viewName = GetViewName(wallName)
    
    If viewName = "" Then
    
        MsgBox "View Name not selected", vbExclamation, "Not Selected!"
        Unload Me
        Exit Sub
        
    End If
    
    Set swMathUtility = swApp.GetMathUtility

    Dim swViewNormalVector As SldWorks.MathVector
    Set swViewNormalVector = swMathUtility.CreateVector(GetViewVector(viewName))
    
    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\ASSEMBLY DRAWING.drwdot", 0, 0, 0)

    Set swSketchMgr = swDrawing.SketchManager

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)
    Call AddLegendBlocks(swDrawing, swSheet)

    Dim swFrontView As SldWorks.View
    Set swFrontView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), viewName, 0.21593179, 0.19172741, 0)

    Dim IsZChannelExists As Boolean
    Dim ViewWidth As Double
    Dim ViewHeight As Double
    Dim MaxHeightComp As IComp
    Dim CompList As IArrListObject

    Dim cChannelList As IArrListObject
    Set cChannelList = New IArrListObject

    Dim zChannelList As IArrListObject
    Set zChannelList = New IArrListObject
    
    Dim lAngleList As IArrListObject
    Set lAngleList = New IArrListObject

    Set CompList = GetComponentsSortedWithYPosition(swFrontView, swDrawing, swViewNormalVector, ViewWidth, _
                ViewHeight, MaxHeightComp, IsZChannelExists, zChannelList, cChannelList, lAngleList)

    Dim IsMultipleAssembly As Boolean
    IsMultipleAssembly = CheckForMultipleAssembly(ViewWidth / swFrontView.ScaleDecimal, ViewHeight / swFrontView.ScaleDecimal)

    Dim subAssyEndComponents As Variant
    If IsMultipleAssembly Then

        Call ActivateDrawingDocument(swTopLevelModel)
        SubAssyForm.Show vbModeless

        IsSubAssyFormClicked = False
        Do While IsSubAssyFormClicked = False

            DoEvents

        Loop

        subAssyEndComponents = GetSelectedComponents
        Call ActivateDrawingDocument(swDrawing)

    End If

    Dim swBottomView As SldWorks.View
    Set swBottomView = ScaleAndInsertBottomView(swDrawing, swFrontView, ViewWidth, ViewHeight)

    Dim FlatCompList As Variant
    Dim DetailedCompList As Variant
    Dim MaxCompHeight As Double
    DetailedCompList = GetComponentsSortedWithXPosition(CompList.Items, FlatCompList, swFrontView, MaxCompHeight)

    Dim vConsolidatedList As Variant

    Dim DoorOrHVACList As IArrListObject
    Set DoorOrHVACList = New IArrListObject

    vConsolidatedList = GetConsolidatedList(DetailedCompList, DoorOrHVACList)

    Set zChannelList = GetChannelCompsWithPos(zChannelList, swFrontView)
    Set cChannelList = GetChannelCompsWithPos(cChannelList, swFrontView)
    Set lAngleList = GetChannelCompsWithPos(lAngleList, swFrontView)

    Call CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList, zChannelList, True)
    Call CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList, cChannelList)
    Call CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList, lAngleList, IsLAngle:=True)

    swDrawing.ActivateView swFrontView.Name

    Dim IsMakeUpExists As Boolean
    Dim subAssyCompDict As Scripting.Dictionary
    Set subAssyCompDict = AddSubAssyComponentsToDictionary(subAssyEndComponents)
    
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
    
    Call AddCallouts(vConsolidatedList, swDrawing, swFrontView, MaxCompHeight, IsMakeUpExists, subAssyCompDict)

    Dim Is12GAPanelExists As Boolean
    Dim IsAllPanels12GA As Boolean
    Is12GAPanelExists = Add12GACircles(FlatCompList, swDrawing, swBottomView, wallName, IsAllPanels12GA)

    Call UpdateBottomViewPosition(FlatCompList, swDrawing, swBottomView)

    Dim swLeftEdge As SldWorks.Edge
    Dim swRightEdge As SldWorks.Edge

    Dim swBottomEdge As SldWorks.Edge
    Set swBottomEdge = AddDimensionInFrontView(swFrontView, FlatCompList, DetailedCompList, MaxHeightComp, swDrawing, swLeftEdge, swRightEdge)

    Dim FlatCompDict As Scripting.Dictionary
    Dim CompNoDict As New Scripting.Dictionary
    Set FlatCompDict = GetCompDictionary(FlatCompList, CompNoDict)

    Dim subAssylist As IArrListObject
    Set subAssylist = New IArrListObject

    If Not IsEmpty(subAssyEndComponents) Then

        Dim vSubAssyComponentsIdx As Variant
        vSubAssyComponentsIdx = GetSubAssyComponentsIndexSorted(subAssyEndComponents, CompNoDict)

        Set subAssylist = AddSplitLines(vSubAssyComponentsIdx, swDrawing, swFrontView, FlatCompDict, CompNoDict, True, swLeftEdge, swRightEdge, False)
        Call AddSplitLines(vSubAssyComponentsIdx, swDrawing, swBottomView, FlatCompDict, CompNoDict, False, swLeftEdge, swRightEdge)

        Call CheckAndAddDoorOrHVACAssy(subAssylist, DoorOrHVACList, CompNoDict)


    End If

    Dim oSubAssy As ISubAssy
    Set oSubAssy = New ISubAssy

    Set oSubAssy.StartComp = FlatCompDict.Items(0)
    Set oSubAssy.EndComp = FlatCompDict.Items(UBound(FlatCompDict.Items))
    Set oSubAssy.StartEdge = swLeftEdge
    Set oSubAssy.EndEdge = swRightEdge
    Set oSubAssy.BottomEdge = swBottomEdge
    
    oSubAssy.StartIdx = 0
    oSubAssy.EndIdx = UBound(FlatCompDict.Items)
    Call oSubAssy.AddDoororHVACList(DoorOrHVACList)

    subAssylist.AddtoList oSubAssy
    
    Dim Countourlist As IArrListObject
    Set Countourlist = AddCrossMarkForAssyCuts(FlatCompDict.Items, swFrontView, swDrawing, oSubAssy)
    
    Call AddCrossMarkForDoor(oSubAssy, swFrontView, swDrawing)
    
    Dim UniqueHVACDict As Scripting.Dictionary
    Set UniqueHVACDict = AddCrossMarkForHVAC(oSubAssy, swFrontView, swDrawing)

    Dim NoteCount As Integer
    Dim AssyNoteNo As Integer
    Call AddStructuralNotes(swDrawing, swSheet, Is12GAPanelExists, IsAllPanels12GA, IsZChannelExists, NoteCount, wallName, Countourlist.Count)
    
    Dim IsSectionViewNeeded As Boolean
    IsSectionViewNeeded = False
    Dim GapForSection As Double
        
    If oSubAssy.GetWidth <= (15.75 - 2.5 * (UBound(UniqueHVACDict.Items) + 1)) * 0.0254 Then
            
        IsSectionViewNeeded = True
        GapForSection = (15.75 * 0.0254 - oSubAssy.GetWidth) / 2
        
    End If
        
    Dim MaxClearance As Double
    Call AddDimensionsForDoororHVACInEachSubAssy(subAssylist, swDrawing, swFrontView, MaxClearance, IsSectionViewNeeded)
    Call AddDimensionNames(subAssylist, wallName, swFrontView)
    Call AddVerticalDimensionsForDoor(oSubAssy.GetDoorAssemblies, swFrontView, swDrawing, NoteCount)

    Call AddVerticalDimensionsForHVAC(UniqueHVACDict.Items, swFrontView, swDrawing, oSubAssy, IsSectionViewNeeded, GapForSection)

    Call SketchLineForNonCornerPanels(swFrontView, wallName, swDrawing, oSubAssy, NoteCount, swBottomEdge, MaxClearance)
    Call CleanUpActivateAndAddViewLabel(swDrawing, swFrontView, wallName, oSubAssy.StartComp.yMin - MaxClearance - 0.0075, (oSubAssy.StartComp.xMin + oSubAssy.EndComp.xMax) / 2)
    
    Call UpdateFrontViewPosition(FlatCompDict.Items, swDrawing, swFrontView)

    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
    
    Unload Me

End Sub

Private Sub AddVerticalDimensionsForHVAC(vHVACItems As Variant, swView As SldWorks.View, _
        swDrawing As SldWorks.DrawingDoc, oSubAssy As ISubAssy, IsSectionViewNeeded As Boolean, GapForSection As Double)

    Const LeftBorderPoint  As Double = 0.01590679
    If Not IsEmpty(vHVACItems) Then
    
        Dim i As Integer
        For i = LBound(vHVACItems) To UBound(vHVACItems)
        
            Dim HVACArrList As IArrListObject
            Set HVACArrList = vHVACItems(i)
            
            Dim FirstHVACAssy As IDoorOrHVACAssy
            Set FirstHVACAssy = HVACArrList.Items(0)
            
            Dim oStartComp As IComp
            Set oStartComp = FirstHVACAssy.StartComp
                
            Dim swHVACBottomEdge As SldWorks.Edge
            Set swHVACBottomEdge = GetEdgeInView(oStartComp, swView, True, False)
            
            Dim swDisplayDim As SldWorks.DisplayDimension

            Dim vCChannelItems As Variant
            vCChannelItems = FirstHVACAssy.cChannelCompList.Items
            
            Dim vLAngleItems As Variant
            vLAngleItems = FirstHVACAssy.lAngleComplist.Items
            
            Dim VerticalSectionView As SldWorks.View
            Dim HorSectionView As SldWorks.View
            
            Dim VerticalSectionOutline As Variant

            Dim j As Integer
            Dim PrevEdge As SldWorks.Edge
            Dim ViewToAddDimension As SldWorks.View
            Dim DimXPos As Double
            Dim Qty As Integer
            
            For j = LBound(vCChannelItems) To UBound(vCChannelItems)
            
                Dim oChannelComp As IComp
                Set oChannelComp = vCChannelItems(j)
                
                Dim oLAngleComp As IComp
                Set oLAngleComp = vLAngleItems(j)
                
                Dim cChannelTopEdge As SldWorks.Edge
                Dim lAngleBottomEdge As SldWorks.Edge
                
                Dim swSketchManager As SldWorks.SketchManager
                Set swSketchManager = swDrawing.SketchManager

                If j = 0 Then

                    Set cChannelTopEdge = GetEdgeInView(oChannelComp, swView, True, True)
                    Set lAngleBottomEdge = GetEdgeInView(oLAngleComp, swView, True, False)
                    
                    Set swDisplayDim = SelectAndAddDimension(cChannelTopEdge, swHVACBottomEdge, swDrawing, _
                                (oStartComp.xMin + oStartComp.xMax) / 2, oStartComp.yMin + 0.01, swView, False)
                    Call AddQtyToDimension(swDisplayDim, HVACArrList.Count)

                    If IsSectionViewNeeded Then
                    
                        swDrawing.ActivateView swView.Name
                        swView.FocusLocked = True
                        
                        Dim VerticalLowerPoint(2) As Double
                        VerticalLowerPoint(0) = 0.75 * oChannelComp.xMin + 0.25 * oChannelComp.xMax
                        VerticalLowerPoint(1) = oChannelComp.yMin - 10 * swView.ScaleDecimal * 0.0254
                        VerticalLowerPoint(2) = 0
                        
                        Dim vVerticalLowerPoint As Variant
                        vVerticalLowerPoint = GetSheetPointInViewSpace(swView, VerticalLowerPoint)
            
                        Dim VerticalUpperPoint(2) As Double
                        VerticalUpperPoint(0) = VerticalLowerPoint(0)
                        VerticalUpperPoint(1) = FirstHVACAssy.lAngleComplist.Items(UBound(FirstHVACAssy.lAngleComplist.Items)).yMax + 10 * swView.ScaleDecimal * 0.0254
                        VerticalUpperPoint(2) = 0
                        
                        Dim vVerticalUpperPoint As Variant
                        vVerticalUpperPoint = GetSheetPointInViewSpace(swView, VerticalUpperPoint)
                        
                        Dim swSketchSegment As SketchSegment
                        Set swSketchSegment = swSketchManager.CreateLine(vVerticalLowerPoint(0), vVerticalLowerPoint(1), vVerticalLowerPoint(2), _
                                            vVerticalUpperPoint(0), vVerticalUpperPoint(1), vVerticalUpperPoint(2))
                                            
                        swSketchSegment.Select4 False, Nothing
                        
                        Dim vExcludedComps As Variant
                        Set VerticalSectionView = swDrawing.CreateSectionViewAt5(LeftBorderPoint + ((i + 1) * GapForSection / (UBound(vHVACItems) + 2)), (VerticalLowerPoint(1) + VerticalUpperPoint(1)) / 2, 0, "A", swCreateSectionViewAtOptions_e.swCreateSectionView_Partial + _
                                            swCreateSectionViewAtOptions_e.swCreateSectionView_NotAligned, vExcludedComps, 0.005)
                                      

                        VerticalSectionView.GetSection.Layer = "FORMAT"
                                      
                        Dim HorizontalLeftPoint(2) As Double
                        HorizontalLeftPoint(0) = FirstHVACAssy.StartComp.xMax - 4 * swView.ScaleDecimal * 0.0254
                        HorizontalLeftPoint(1) = (oChannelComp.yMax + oLAngleComp.yMin) / 2
                        HorizontalLeftPoint(2) = 0
                        
                        Dim vHorizontalLeftPoint As Variant
                        vHorizontalLeftPoint = GetSheetPointInViewSpace(swView, HorizontalLeftPoint)
                        
                        Dim HorizontalRightPoint(2) As Double
                        HorizontalRightPoint(0) = FirstHVACAssy.EndComp.xMin + 4 * swView.ScaleDecimal * 0.0254
                        HorizontalRightPoint(1) = HorizontalLeftPoint(1)
                        HorizontalRightPoint(2) = 0
                        
                        Dim vHorizontalRightPoint As Variant
                        vHorizontalRightPoint = GetSheetPointInViewSpace(swView, HorizontalRightPoint)
                        
                        Set swSketchSegment = swSketchManager.CreateLine(vHorizontalRightPoint(0), vHorizontalRightPoint(1), vHorizontalRightPoint(2), _
                                            vHorizontalLeftPoint(0), vHorizontalLeftPoint(1), vHorizontalLeftPoint(2))
                                            
                        swSketchSegment.Select4 False, Nothing
                        
                        Set HorSectionView = swDrawing.CreateSectionViewAt5(LeftBorderPoint + ((i + 1) * GapForSection / (UBound(vHVACItems) + 2)), oSubAssy.StartComp.yMin - 0.02, 0, "B", swCreateSectionViewAtOptions_e.swCreateSectionView_Partial + _
                                            swCreateSectionViewAtOptions_e.swCreateSectionView_NotAligned, vExcludedComps, 0.005)
                                            
                        HorSectionView.GetSection.Layer = "FORMAT"
                        
                        If VerticalSectionView Is Nothing Then
                            
                            Set ViewToAddDimension = swView
                            DimXPos = (oStartComp.xMin + oStartComp.xMax) / 2
                            Qty = HVACArrList.Count
                              
                        Else
                        
                            VerticalSectionOutline = VerticalSectionView.GetOutline
                            Set ViewToAddDimension = VerticalSectionView
                            DimXPos = VerticalSectionOutline(0) - 0.005
                            
                            Set cChannelTopEdge = GetEdgeInView(oChannelComp, VerticalSectionView, True, True, IsSection:=IsSectionViewNeeded)
                            Set lAngleBottomEdge = GetEdgeInView(oLAngleComp, VerticalSectionView, True, False, IsSection:=IsSectionViewNeeded)
                            Qty = 1
                            
                            Call UpdateSectionLabel(VerticalSectionView, HVACArrList.Count)

                        End If
                        
                        If Not HorSectionView Is Nothing Then
                        
                            Dim HorOutline As Variant
                            HorOutline = HorSectionView.GetOutline

                            Dim LeftCompEdge As SldWorks.Edge
                            Set LeftCompEdge = GetEdgeInView(oStartComp, HorSectionView, False, True, IsSection:=IsSectionViewNeeded)
                            
                            Dim RightCompEdge As SldWorks.Edge
                            Set RightCompEdge = GetEdgeInView(FirstHVACAssy.EndComp, HorSectionView, False, False, IsSection:=IsSectionViewNeeded)
                            
                            Set swDisplayDim = SelectAndAddDimension(LeftCompEdge, RightCompEdge, swDrawing, _
                                (HorOutline(0) + HorOutline(2)) / 2, HorOutline(3) + 0.0025, HorSectionView, False)
                                
                            Call UpdateSectionLabel(HorSectionView, HVACArrList.Count)
                           
                        End If
                        
                        
                    Else
                    
                        Set ViewToAddDimension = swView
                        DimXPos = (oStartComp.xMin + oStartComp.xMax) / 2
                        Qty = HVACArrList.Count
                        
                
                    End If
                
                Else
                    
                    Set cChannelTopEdge = GetEdgeInView(oChannelComp, ViewToAddDimension, True, True, IsSection:=IsSectionViewNeeded)
                    Set lAngleBottomEdge = GetEdgeInView(oLAngleComp, ViewToAddDimension, True, False, IsSection:=IsSectionViewNeeded)

                    Set swDisplayDim = SelectAndAddDimension(cChannelTopEdge, PrevEdge, swDrawing, _
                                DimXPos, oChannelComp.yMin, ViewToAddDimension, False)
                                
                    Call AddQtyToDimension(swDisplayDim, Qty)
                    
                End If
                
                Set swDisplayDim = SelectAndAddDimension(lAngleBottomEdge, cChannelTopEdge, swDrawing, _
                            DimXPos, oChannelComp.yMax + 0.001, ViewToAddDimension, False)
                                    
                Call AddQtyToDimension(swDisplayDim, Qty)
                Set PrevEdge = lAngleBottomEdge
                Call SelectAndAddAnnotationForEdge(oChannelComp.GetComponent, cChannelTopEdge, swDrawing, ViewToAddDimension, Qty, 0.0025, -0.0075)
                Call SelectAndAddAnnotationForEdge(oLAngleComp.GetComponent, lAngleBottomEdge, swDrawing, ViewToAddDimension, Qty, 0.0025)
  
            Next j

        Next i


    End If

End Sub

Sub UpdateSectionLabel(swView As SldWorks.View, Qty As Integer)
    
    If Qty > 1 Then
    
        Dim swNote As SldWorks.Note
        Set swNote = swView.GetFirstNote
                                
        swNote.SetText "<VLNAME> <VLLABEL>" & vbCrLf & "<FONT size=8PTS style=R>TYP. @ " & Qty & " PLACES"
    
    End If

End Sub

Function SelectAndAddAnnotationForEdge(swComp As SldWorks.Component2, swEdge As SldWorks.Edge, swDrawing As SldWorks.DrawingDoc, _
                swView As SldWorks.View, Qty As Integer, Optional XClearance As Double = 0, _
                        Optional YClearance As Double = 0.0075, Optional PercentageFromStart As Double = 0.5, _
                            Optional BalloonStyle As swBalloonStyle_e = swBalloonStyle_e.swBS_Inspection) As SldWorks.Annotation

    Dim IsSelected As Boolean
    Dim SelXPos As Double
    Dim SelYPos As Double
    IsSelected = SelectEdgeWithSelectData(swEdge, swView, swDrawing, swComp, SelXPos, SelYPos, PercentageFromStart)
    
    If IsSelected Then

        Dim swNote As SldWorks.Note
        Set SelectAndAddAnnotationForEdge = InsertBalloonAndGetAnnotations(swDrawing, Qty, SelXPos + XClearance, SelYPos + YClearance, BalloonStyle)
        
    End If

End Function

Function SelectEdgeWithSelectData(swEdge As SldWorks.Edge, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
                swComp As SldWorks.Component2, ByRef SelXPos As Double, ByRef SelYPos As Double, Optional PercentageFromStart As Double = 0.5) As Boolean

    Dim swSelectMgr As SldWorks.SelectionMgr
    Set swSelectMgr = swDrawing.SelectionManager
    
    Dim swSelectData As SldWorks.SelectData
    Set swSelectData = swSelectMgr.CreateSelectData
    
    Dim vStartPoint As Variant
    vStartPoint = swEdge.GetStartVertex.GetPoint
    vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
    
    Dim vEndPoint As Variant
    vEndPoint = swEdge.GetEndVertex.GetPoint
    vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)
    
    Dim swMathStartPoint As SldWorks.MathPoint
    Set swMathStartPoint = swMathUtility.CreatePoint(vStartPoint)
    
    Dim swMathEndPoint As SldWorks.MathPoint
    Set swMathEndPoint = swMathUtility.CreatePoint(vEndPoint)
    
    Dim swPosVector As SldWorks.MathVector
    Set swPosVector = swMathEndPoint.Subtract(swMathStartPoint)
    
    Set swMathStartPoint = swMathStartPoint.AddVector(swPosVector.Scale(PercentageFromStart))
    
    SelXPos = swMathStartPoint.ArrayData(0)
    SelYPos = swMathStartPoint.ArrayData(1)
    
    swSelectData.View = swView
    swSelectData.X = SelXPos '(vStartPoint(0) + vEndPoint(0)) / 2
    swSelectData.Y = SelYPos 'vStartPoint(1)
    
    Dim swEntity As SldWorks.Entity
    Set swEntity = swEdge

    SelectEdgeWithSelectData = swEntity.Select4(False, swSelectData)
    
End Function

Sub AddQtyToDimension(swDisplayDim As SldWorks.DisplayDimension, Qty As Integer)

    If Qty > 1 Then
                                    
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X "
                            
    End If
   
End Sub

Sub AddCrossMarkForDoor(oSubAssy As ISubAssy, swView As SldWorks.View, _
                swDrawing As SldWorks.DrawingDoc)
                
    Dim vDoorAssy As Variant
    vDoorAssy = oSubAssy.GetDoorAssemblies
    
    If Not IsEmpty(vDoorAssy) Then
    
        swDrawing.ActivateSheet swDrawing.GetCurrentSheet.GetName
        swDrawing.ActivateView swView.Name
        
        swView.FocusLocked = True
    
        Dim i As Integer
        For i = LBound(vDoorAssy) To UBound(vDoorAssy)
            
            Dim oDoorAssy As IDoorOrHVACAssy
            Set oDoorAssy = vDoorAssy(i)
            
            If oDoorAssy.cChannelCompList.Count = 1 Then
            
                Dim DoorLeftEdge As SldWorks.Edge
                Set DoorLeftEdge = GetEdgeInView(oDoorAssy.StartComp, swView, False, True)
                
                Dim DoorRightEdge As SldWorks.Edge
                Set DoorRightEdge = GetEdgeInView(oDoorAssy.EndComp, swView, False, False)
                
                Dim DoorBottomEdge As SldWorks.Edge
                Set DoorBottomEdge = GetEdgeInView(oDoorAssy.StartComp, swView, True, False)
                
                Dim cChannelComp As IComp
                Set cChannelComp = oDoorAssy.cChannelCompList.Items(0)
                
                Dim DoorTopEdge As SldWorks.Edge
                Set DoorTopEdge = GetEdgeInView(cChannelComp, swView, True, False)

                Dim LowerLeftPoint(2) As Double
                LowerLeftPoint(0) = oDoorAssy.StartComp.xMax
                LowerLeftPoint(1) = oDoorAssy.StartComp.yMin
                LowerLeftPoint(2) = 0
                
                Dim vLowerLeftPoint As Variant
                vLowerLeftPoint = GetSheetPointInViewSpace(swView, LowerLeftPoint)
    
                Dim LowerRightPoint(2) As Double
                LowerRightPoint(0) = oDoorAssy.EndComp.xMin
                LowerRightPoint(1) = oDoorAssy.StartComp.yMin
                LowerRightPoint(2) = 0
                
                Dim vLowerRightPoint As Variant
                vLowerRightPoint = GetSheetPointInViewSpace(swView, LowerRightPoint)
    
                Dim UpperLeftPoint(2) As Double
                UpperLeftPoint(0) = oDoorAssy.StartComp.xMax
                UpperLeftPoint(1) = cChannelComp.yMin
                UpperLeftPoint(2) = 0
                
                Dim vUpperLeftPoint As Variant
                vUpperLeftPoint = GetSheetPointInViewSpace(swView, UpperLeftPoint)
    
                Dim UpperRightPoint(2) As Double
                UpperRightPoint(0) = oDoorAssy.EndComp.xMin
                UpperRightPoint(1) = cChannelComp.yMin
                UpperRightPoint(2) = 0
                
                Dim vUpperRightPoint As Variant
                vUpperRightPoint = GetSheetPointInViewSpace(swView, UpperRightPoint)
                
                Dim swSketchManager As SldWorks.SketchManager
                Set swSketchManager = swDrawing.SketchManager
                
                Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerLeftPoint, vUpperRightPoint, DoorLeftEdge, DoorRightEdge, DoorBottomEdge, DoorTopEdge)
                Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerRightPoint, vUpperLeftPoint, DoorRightEdge, DoorLeftEdge, DoorBottomEdge, DoorTopEdge)
            
            End If
        Next i
        
    End If

End Sub

Function AddCrossMarkForHVAC(oSubAssy As ISubAssy, swView As SldWorks.View, _
                swDrawing As SldWorks.DrawingDoc) As Scripting.Dictionary
                
    Dim vHVACAssy As Variant
    vHVACAssy = oSubAssy.GetHVACAssemblies
    
    Set AddCrossMarkForHVAC = New Scripting.Dictionary
    
    If Not IsEmpty(vHVACAssy) Then
    
        swDrawing.ActivateSheet swDrawing.GetCurrentSheet.GetName
        swDrawing.ActivateView swView.Name
        
        swView.FocusLocked = True
    
        Dim i As Integer
        For i = LBound(vHVACAssy) To UBound(vHVACAssy)
            
            Dim oHVACAssy As IDoorOrHVACAssy
            Set oHVACAssy = vHVACAssy(i)

            If oHVACAssy.cChannelCompList.Count = oHVACAssy.lAngleComplist.Count Then
            
                oHVACAssy.lAngleComplist.SortItems "yMin", False
                oHVACAssy.cChannelCompList.SortItems "yMin", False
            
                Dim HVACLeftEdge As SldWorks.Edge
                Set HVACLeftEdge = GetEdgeInView(oHVACAssy.StartComp, swView, False, True)
                
                Dim HVACRightEdge As SldWorks.Edge
                Set HVACRightEdge = GetEdgeInView(oHVACAssy.EndComp, swView, False, False)

                Dim vChannelItems As Variant
                vChannelItems = oHVACAssy.cChannelCompList.Items
                
                Dim vLAngleItems As Variant
                vLAngleItems = oHVACAssy.lAngleComplist.Items
                
                If Not IsEmpty(vLAngleItems) And Not IsEmpty(vChannelItems) Then
                    
                    Dim keyVal As Double
                    keyVal = Round(vChannelItems(0).yMax, 4)
                    
                    If AddCrossMarkForHVAC.Exists(keyVal) Then
                        
                        AddCrossMarkForHVAC.Item(keyVal).AddtoList oHVACAssy

                    Else
                    
                        Dim ArrList As IArrListObject
                        Set ArrList = New IArrListObject
                        
                        ArrList.AddtoList oHVACAssy
                        AddCrossMarkForHVAC.Add keyVal, ArrList

                    End If

                    Dim j As Integer
                    For j = LBound(vLAngleItems) To UBound(vLAngleItems)
                    
                        Dim cChannelComp As IComp
                        Set cChannelComp = vChannelItems(j)
                        
                        Dim lAngleComp As IComp
                        Set lAngleComp = vLAngleItems(j)
                        
                        If cChannelComp.yMax < lAngleComp.yMin Then

                            Dim HVACBottomEdge As SldWorks.Edge
                            Set HVACBottomEdge = GetEdgeInView(cChannelComp, swView, True, True)
                            
                            Dim HVACTopEdge As SldWorks.Edge
                            Set HVACTopEdge = GetEdgeInView(lAngleComp, swView, True, False)
                            
                            Dim LowerLeftPoint(2) As Double
                            LowerLeftPoint(0) = oHVACAssy.StartComp.xMax
                            LowerLeftPoint(1) = cChannelComp.yMax
                            LowerLeftPoint(2) = 0
                            
                            Dim vLowerLeftPoint As Variant
                            vLowerLeftPoint = GetSheetPointInViewSpace(swView, LowerLeftPoint)
                
                            Dim LowerRightPoint(2) As Double
                            LowerRightPoint(0) = oHVACAssy.EndComp.xMin
                            LowerRightPoint(1) = cChannelComp.yMax
                            LowerRightPoint(2) = 0
                            
                            Dim vLowerRightPoint As Variant
                            vLowerRightPoint = GetSheetPointInViewSpace(swView, LowerRightPoint)
                
                            Dim UpperLeftPoint(2) As Double
                            UpperLeftPoint(0) = oHVACAssy.StartComp.xMax
                            UpperLeftPoint(1) = lAngleComp.yMin
                            UpperLeftPoint(2) = 0
                            
                            Dim vUpperLeftPoint As Variant
                            vUpperLeftPoint = GetSheetPointInViewSpace(swView, UpperLeftPoint)
                
                            Dim UpperRightPoint(2) As Double
                            UpperRightPoint(0) = oHVACAssy.EndComp.xMin
                            UpperRightPoint(1) = lAngleComp.yMin
                            UpperRightPoint(2) = 0
                            
                            Dim vUpperRightPoint As Variant
                            vUpperRightPoint = GetSheetPointInViewSpace(swView, UpperRightPoint)
                            
                            Dim swSketchManager As SldWorks.SketchManager
                            Set swSketchManager = swDrawing.SketchManager
                            
                            Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerLeftPoint, vUpperRightPoint, HVACLeftEdge, HVACRightEdge, HVACBottomEdge, HVACTopEdge)
                            Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerRightPoint, vUpperLeftPoint, HVACRightEdge, HVACLeftEdge, HVACBottomEdge, HVACTopEdge)
                            
                        End If

                    Next j
                    
                End If

            End If
            
        Next i
        
    End If

End Function

Function AddCrossMarkForAssyCuts(vComps As Variant, swView As SldWorks.View, _
                swDrawing As SldWorks.DrawingDoc, oSubAssy As ISubAssy) As IArrListObject
                
    If Not IsEmpty(vComps) Then
    
        Dim FullAssyName As String
        FullAssyName = Replace(vComps(0).GetComponent.Name2, "/" & _
                    Right(vComps(0).GetComponent.Name2, Len(vComps(0).GetComponent.Name2) - InStrRev(vComps(0).GetComponent.Name2, "/")), "")
        
        Dim AssyName As String
        AssyName = Right(FullAssyName, Len(FullAssyName) - InStrRev(FullAssyName, "/"))
                    
        Dim swTopLevelAssy As SldWorks.AssemblyDoc
        Set swTopLevelAssy = swTopLevelModel
        
        Dim swWallComp As SldWorks.Component2
        Set swWallComp = swTopLevelAssy.GetComponentByName(AssyName)

        Dim swWallAssy As SldWorks.AssemblyDoc
        Set swWallAssy = swWallComp.GetModelDoc2()
        
        Dim Errors As Long
        swApp.ActivateDoc3 swWallAssy.GetPathName, True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Errors
        
        Dim vAssyCutFeatures As Variant
        vAssyCutFeatures = GetAssyCutFeaturesIfAny(vComps, swWallAssy)
          
        Set AddCrossMarkForAssyCuts = GetContoursAndAddCrossMark(vAssyCutFeatures, swDrawing, swView, FullAssyName, swWallComp, oSubAssy)

        swApp.CloseDoc swWallAssy.GetPathName

    End If

End Function

Sub CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, FirstPoint As Variant, SecondPoint As Variant, FirstPtVerticalEdge As SldWorks.Edge, _
                SecondPtVerticalEdge As SldWorks.Edge, FirstPtHorEdge As SldWorks.Edge, SecondPtHorEdge As SldWorks.Edge)
    
    
    
    Dim swSketchSegment As SketchSegment
    Set swSketchSegment = swSketchManager.CreateLine(FirstPoint(0), FirstPoint(1), FirstPoint(2), _
                        SecondPoint(0), SecondPoint(1), SecondPoint(2))
    swSketchSegment.ConstructionGeometry = True
    
    If Not swSketchSegment Is Nothing Then
        
        Dim swSketchLine As SldWorks.sketchLine
        Set swSketchLine = swSketchSegment
        
        Dim swFirstPoint As SldWorks.sketchPoint
        Set swFirstPoint = swSketchLine.GetStartPoint2
        
        Call AddCoincidentRelationbwPointAndEdge(FirstPtVerticalEdge, swFirstPoint, swDrawing, swView)
        Call AddCoincidentRelationbwPointAndEdge(FirstPtHorEdge, swFirstPoint, swDrawing, swView)
        
        
        Dim swSecondPoint As SldWorks.sketchPoint
        Set swSecondPoint = swSketchLine.GetEndPoint2
        
        Call AddCoincidentRelationbwPointAndEdge(SecondPtVerticalEdge, swSecondPoint, swDrawing, swView)
        Call AddCoincidentRelationbwPointAndEdge(SecondPtHorEdge, swSecondPoint, swDrawing, swView)

    End If
    
End Sub

Sub AddCoincidentRelationbwPointAndEdge(swEdge As SldWorks.Edge, swSketchPoint As SldWorks.sketchPoint, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    swView.SelectEntity swEdge, False
    swSketchPoint.Select4 True, Nothing
    
    swDrawing.SketchAddConstraints "sgCOINCIDENT"
    
    
End Sub


Function GetAssyCutFeaturesIfAny(vComps As Variant, swWallAssy As SldWorks.AssemblyDoc) As Variant

    Dim CutFeaturesDict As Scripting.Dictionary
    Set CutFeaturesDict = New Scripting.Dictionary

    Dim swFeatManager As SldWorks.FeatureManager
    Set swFeatManager = swWallAssy.FeatureManager
    
    Dim vFeats As Variant
    vFeats = swFeatManager.GetFeatures(True)
    
    Dim i As Integer
    For i = UBound(vFeats) To LBound(vFeats) Step -1
    
        Dim swFeat As SldWorks.Feature
        Set swFeat = vFeats(i)
        
        Debug.Print swFeat.Name
        Debug.Print swFeat.GetTypeName2
        
        If False = swFeat.IsSuppressed Then
        
            If swFeat.GetTypeName2 = "Cut" Then
    
                If IsFeatureAffectAnyComp(vComps, swWallAssy, swFeat) Then
            
                    If Not CutFeaturesDict.Exists(swFeat.Name) Then
                    
                        CutFeaturesDict.Add swFeat.Name, swFeat
                        
                    End If
                    
                End If
                
            End If
            
        End If
        
        If swFeat.GetTypeName2 = "MateGroup" Then
        
            Exit For
            
        End If
        
    Next i
    
    GetAssyCutFeaturesIfAny = CutFeaturesDict.Items

End Function

Function IsFeatureAffectAnyComp(vComps As Variant, _
            swWallAssy As SldWorks.AssemblyDoc, swFeat As SldWorks.Feature) As Boolean
            
    IsFeatureAffectAnyComp = False
      
    Dim vAffectedComps As Variant
    vAffectedComps = swWallAssy.GetFeatureScope(swFeat)
    
    Dim i As Integer
    For i = LBound(vAffectedComps) To UBound(vAffectedComps)
    
        Dim swFeatAffectedComp As SldWorks.Component2
        Set swFeatAffectedComp = vAffectedComps(i)
        
        Dim swCompModel As SldWorks.ModelDoc2
        Set swCompModel = swFeatAffectedComp.GetModelDoc2()
        
        If Not swCompModel Is Nothing Then
 
        Dim swCompProp As SldWorks.CustomPropertyManager
        Set swCompProp = swCompModel.Extension.CustomPropertyManager("")
                
        Dim Profile As String
        Dim ValOut As String
        Dim wasResolved As Boolean
        swCompProp.Get5 "Profile", False, ValOut, Profile, wasResolved
        
        If InStr(Profile, "EXT-") > 0 Then
        
            Dim j As Integer
            For j = LBound(vComps) To UBound(vComps)
            
                Dim swComp As SldWorks.Component2
                Set swComp = vComps(j).GetComponent
            
                If InStr(swComp.Name2, swFeatAffectedComp.Name2) > 0 Then
                    
                    IsFeatureAffectAnyComp = True
                    Exit Function
                    
                End If
                
            Next j
            
        End If
        
        End If

    Next i

End Function

Function GetContoursAndAddCrossMark(vCutFeatures As Variant, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, AssyName As String, swComp As SldWorks.Component2, oSubAssy As ISubAssy) As IArrListObject
        
    Set GetContoursAndAddCrossMark = New IArrListObject
    
    If Not IsEmpty(vCutFeatures) Then
    
        Dim i As Integer
        For i = LBound(vCutFeatures) To UBound(vCutFeatures)
        
            Dim swFeat As SldWorks.Feature
            Set swFeat = vCutFeatures(i)
            
            Dim swSubFeat As SldWorks.Feature
            Set swSubFeat = swFeat.GetFirstSubFeature
            
            If swSubFeat.GetTypeName2 = "ProfileFeature" Then
            
                Dim swSketch As SldWorks.Sketch
                Set swSketch = swSubFeat.GetSpecificFeature2
                
                Dim vContours As Variant
                vContours = GetSketchContours(swSketch, swComp, swView)

                If Not (IsEmpty(vContours)) Then
                
                    Call GetContoursAndAddCrossMark.AddItems(vContours)
                    Call AddCrossMarkAndDimensionsForContours(vContours, swDrawing, swSubFeat, swSketch, swView, AssyName, oSubAssy)
                    
                End If
            
            End If

        Next i
        
    End If
    
End Function


Private Sub UpdateFrontViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim oStartComp As IComp
    Set oStartComp = vComps(0)
    
    Dim oEndComp As IComp
    Set oEndComp = vComps(UBound(vComps))
    
    Dim CenterX As Double
    CenterX = (oStartComp.xMin + oEndComp.xMax) / 2

    Dim viewPosition As Variant
    viewPosition = swView.Position

    viewPosition(0) = viewPosition(0) + (viewPosition(0) - CenterX)

    swView.Position = viewPosition
    
End Sub


Private Sub AddVerticalDimensionsForDoor(vDoorOrHVACItems As Variant, swView As SldWorks.View, _
        swDrawing As SldWorks.ModelDoc2, Count As Integer)

    If Not IsEmpty(vDoorOrHVACItems) Then
    
        Dim DoorOrHVACDict As Scripting.Dictionary
        Set DoorOrHVACDict = New Scripting.Dictionary
        
        Dim DoororHVACQtyDict As Scripting.Dictionary
        Set DoororHVACQtyDict = New Scripting.Dictionary
    
        Dim i As Integer
        For i = LBound(vDoorOrHVACItems) To UBound(vDoorOrHVACItems)
        
            Dim oDoorOrHVACAssy As IDoorOrHVACAssy
            Set oDoorOrHVACAssy = vDoorOrHVACItems(i)
            
            Dim oStartComp As IComp
            Set oStartComp = oDoorOrHVACAssy.StartComp
                
            Dim swDoorOrHVACBottomEdge As SldWorks.Edge
            Set swDoorOrHVACBottomEdge = GetEdgeInView(oStartComp, swView, True, False)
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            
            oDoorOrHVACAssy.cChannelCompList.SortItems "yMax", False
            
            Dim vCChannelItems As Variant
            vCChannelItems = oDoorOrHVACAssy.cChannelCompList.Items
            
            Dim vZChannelItems As Variant
            vZChannelItems = oDoorOrHVACAssy.zChannelComplist.Items
            
            If Not IsEmpty(vZChannelItems) Then
            
                If UBound(vZChannelItems) = 0 Then
                
                    Dim zChannelComp As IComp
                    Set zChannelComp = vZChannelItems(0)
                    
                    Dim zChannelEdge As SldWorks.Edge
                    Set zChannelEdge = GetEdgeInView(zChannelComp, swView, True, True)
                    
                    Dim IsSelected As Boolean
                    Dim SelXPos As Double
                    Dim SelYPos As Double
                    IsSelected = SelectEdgeWithSelectData(zChannelEdge, swView, swDrawing, zChannelComp.GetComponent, SelXPos, SelYPos, 0.35)
    
                    If IsSelected Then
                        
                        Dim swStackedBalloonOptions As SldWorks.StackedBalloonOptions
                        Set swStackedBalloonOptions = swDrawing.Extension.CreateStackedBalloonOptions
                        
                        swStackedBalloonOptions.StackDirection = swStackedBalloonDirection_e.swStackedBalloonDir_Up
                        swStackedBalloonOptions.Style = swBalloonStyle_e.swBS_Box
                        swStackedBalloonOptions.Size = swBalloonFit_e.swBF_Tightest
                        swStackedBalloonOptions.UpperTextContent = swBalloonTextContent_e.swBalloonTextCustom
                        swStackedBalloonOptions.UpperText = "1411009"
                        swStackedBalloonOptions.ShowQuantity = False

                        Dim swNote As SldWorks.Note
                        Set swNote = swDrawing.Extension.InsertStackedBalloon2(swStackedBalloonOptions)
                        
                        If Not swNote Is Nothing Then
                            
                            Dim swAnn As SldWorks.Annotation
                            Set swAnn = swNote.GetAnnotation
                            
                            swAnn.SetPosition2 SelXPos, SheetPosForLastBalloon + 0.001, 0
                            
                            If swNote.IsStackedBalloon Then
                                
                                Dim swBalloonStack As SldWorks.BalloonStack
                                Set swBalloonStack = swNote.GetBalloonStack
                                
                                Dim StackedNote As SldWorks.Note
                                Set StackedNote = swBalloonStack.AddTo(swBalloonTextContent_e.swBalloonTextCustom, "Z-CHANNEL ASSEMBLY", swBalloonTextContent_e.swBalloonTextCustom, "")
                            
                            End If
                        
                        End If

                    End If

                End If
            
            End If
                
            If Not IsEmpty(vCChannelItems) Then
            
                Dim oComp As IComp
                Set oComp = vCChannelItems(0)
                
                Dim yDiff As Double

                If oDoorOrHVACAssy.IsDoor Then
            
                    If UBound(vCChannelItems) = 0 Then
                    
                        yDiff = Round(Abs(oComp.yMin - oStartComp.yMin), 3)
                        
                        If Not DoorOrHVACDict.Exists(yDiff) Then
                        
                            Set swDisplayDim = SelectAndAddDimension(GetEdgeInView(oComp, swView, True, False), swDoorOrHVACBottomEdge, swDrawing, _
                            (oStartComp.xMin + oStartComp.xMax) / 2, oStartComp.yMin + 0.01, swView, False)
                            
                            
                            If Not swDisplayDim Is Nothing Then
                            
                                swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "SEE NOTE " & Count - 1
                            
                                DoorOrHVACDict.Add yDiff, swDisplayDim
                                DoororHVACQtyDict.Add yDiff, 1
                                
                            End If
                            
                        Else
                        
                            DoororHVACQtyDict(yDiff) = DoororHVACQtyDict(yDiff) + 1
                            
                        End If
                        
                    End If

                Else
                    
                    yDiff = Round(Abs(oComp.yMax - oStartComp.yMin), 4)
                    
                    If Not DoorOrHVACDict.Exists(yDiff) Then
                    
                        Set swDisplayDim = SelectAndAddDimension(GetEdgeInView(oComp, swView, True, True), swDoorOrHVACBottomEdge, swDrawing, _
                            (oStartComp.xMin + oStartComp.xMax) / 2, oStartComp.yMin + 0.01, swView, False)
                            
                        If Not swDisplayDim Is Nothing Then
                            
                            DoorOrHVACDict.Add yDiff, swDisplayDim
                            DoororHVACQtyDict.Add yDiff, 1
                            
                        End If
                            
                    Else
                        
                         DoororHVACQtyDict(yDiff) = DoororHVACQtyDict(yDiff) + 1
                         
                    End If
            
                End If

            End If

        Next i
        
        Call AddQtyToDimensions(DoorOrHVACDict, DoororHVACQtyDict)

    End If

End Sub

Private Sub AddQtyToDimensions(DoorOrHVACDict As Scripting.Dictionary, DoororHVACQtyDict As Scripting.Dictionary)

    Dim doororHVACKeys As Variant
    doororHVACKeys = DoorOrHVACDict.Keys
    
    If Not IsEmpty(doororHVACKeys) Then
    
        Dim i As Integer
        For i = LBound(doororHVACKeys) To UBound(doororHVACKeys)
        
            If DoororHVACQtyDict.Item(doororHVACKeys(i)) > 1 Then
        
                Dim swDisplayDim As SldWorks.DisplayDimension
                Set swDisplayDim = DoorOrHVACDict.Item(doororHVACKeys(i))
            
            
                swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, DoororHVACQtyDict.Item(doororHVACKeys(i)) & "X "
                
            End If
        
        Next i
        
    End If

End Sub

Private Function GetChannelCompsWithPos(ChannelList As IArrListObject, swView As SldWorks.View) As IArrListObject

    Set GetChannelCompsWithPos = New IArrListObject
    
    If Not IsEmpty(ChannelList.Items) Then
    
        Dim vComps As Variant
        vComps = ChannelList.Items

        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
    
            Dim MinPoint As Variant
            Dim MaxPoint As Variant
            Dim vBodyMinPoint(2) As Double
            Dim vBodyMaxPoint(2) As Double
            
            Dim vNormalFaces As Variant
            
            Dim swComp As SldWorks.Component2
            Set swComp = vComps(i)
            
            Debug.Print swComp.Name2
            
            Call GetMinMaxBodyPointsInSheetSpace(swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, swView, True)
                
            Dim oComp As IComp
            Set oComp = New IComp
            oComp.Initialize swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, vNormalFaces
            
            GetChannelCompsWithPos.AddtoList oComp
            
        Next i
        
        GetChannelCompsWithPos.SortItems "xMin", False
        
    End If

End Function

Private Sub CheckAndAddChannelsToDoorOrHVACList(DoorOrHVACList As IArrListObject, ChannelList As IArrListObject, _
        Optional IsZChannel As Boolean = False, Optional IsLAngle As Boolean = False)
    
    Dim vDoorOrHVACItems As Variant
    vDoorOrHVACItems = DoorOrHVACList.Items
    
    Dim vChannelItems As Variant
    'ChannelList.SortItems "xMin", False
    vChannelItems = ChannelList.Items
    
    Dim i As Integer
    Dim j As Integer
    
    Dim LastSubAssyIdx As Integer
    LastSubAssyIdx = 0
    
    If Not IsEmpty(vChannelItems) And Not IsEmpty(vDoorOrHVACItems) Then
    
        For i = LBound(vChannelItems) To UBound(vChannelItems)
        
            Dim oComp As IComp
            Set oComp = vChannelItems(i)
            
            For j = LastSubAssyIdx To UBound(vDoorOrHVACItems)
            
                Dim oDoorOrHVACAssy As IDoorOrHVACAssy
                Set oDoorOrHVACAssy = vDoorOrHVACItems(j)
    
                If oDoorOrHVACAssy.AddToChannelList(oComp, IsZChannel, IsLAngle) Then
                    
                    LastSubAssyIdx = j
                    Exit For
                    
                End If
                
           Next j
    
        Next i
    
    End If
    
End Sub

Function AddSubAssyComponentsToDictionary(vComps As Variant) As Scripting.Dictionary

    Set AddSubAssyComponentsToDictionary = New Scripting.Dictionary
    
    If Not IsEmpty(vComps) Then
    
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
            
            If Not AddSubAssyComponentsToDictionary.Exists(vComps(i).Name2) Then
            
                AddSubAssyComponentsToDictionary.Add vComps(i).Name2, vComps
                
            End If
        
        Next i
    
    End If
    
End Function

Sub CheckandAddLayer(LayName As String, LayerDesc As String, swLayerMgr As SldWorks.LayerMgr)

    Dim vLayNames As Variant
    vLayNames = swLayerMgr.GetLayerList
    
    Dim IsLayerExists As Boolean
    
    Dim i As Integer
    For i = 0 To UBound(vLayNames)
    
        If vLayNames(i) = LayName Then
        
            IsLayerExists = True
            Exit For
            
        End If
        
    Next i
    
    If Not (IsLayerExists) Then
    
        swLayerMgr.AddLayer LayName, LayerDesc, 0, swLineStyles_e.swLineDEFAULT, swLineWeights_e.swLW_NONE
        
        Dim swLayer As SldWorks.Layer
        Set swLayer = swLayerMgr.GetLayer(LayName)
        
        swLayer.Style = swLineStyles_e.swLineCENTER
        swLayer.Width = swLineWeights_e.swLW_THICK5
        
    End If
    
End Sub

Private Sub SketchLineForNonCornerPanels(swView As SldWorks.View, wallName As String, _
        swDrawing As SldWorks.ModelDoc2, oSubAssy As ISubAssy, CeilingNoteIdx As Integer, swBottomEdge As SldWorks.Edge, ByRef MaxClearance As Double)
    
    swDrawing.ActivateView swView.Name
    
    If InStr(wallName, "Wall") > 0 Then
    
        Dim viewDrawComp As SldWorks.DrawingComponent
        Set viewDrawComp = swView.RootDrawingComponent
        
        Dim viewComp As SldWorks.Component2
        Set viewComp = viewDrawComp.Component
        
        Debug.Print viewComp.Name2
    
        Dim swControlSketch As SldWorks.Component2
        Set swControlSketch = GetControlSketch

        Dim PlaneName As String
        PlaneName = "Ceiling"
        
        swDrawing.Extension.SelectByID2 PlaneName & "@" & viewDrawComp.Name & "@" & swView.Name & "/" & swControlSketch.Name & "@" & viewComp.Name2, "PLANE", 0, 0, 0, False, 0, Nothing, 0
        swView.SelectEntity swBottomEdge, True
        
        Dim swCeilingDim As SldWorks.DisplayDimension
        Set swCeilingDim = swDrawing.AddVerticalDimension2(oSubAssy.EndComp.xMax + 0.01, (oSubAssy.StartComp.yMin + oSubAssy.StartComp.yMax) / 2, 0)
        
        If Not swCeilingDim Is Nothing Then
        
            swCeilingDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "SEE NOTE " & CeilingNoteIdx
            
        End If
        
        Dim swStartSketch As SldWorks.SketchSegment
        Dim swEndSketch As SldWorks.SketchSegment
        
        If Not InStr(oSubAssy.StartComp.GetCustomProperty("Profile"), "CORNER") > 0 Then
        
            PlaneName = GetPlaneName(wallName, True)
            Set swStartSketch = CreateSketchLinesForNonCornerPanels(PlaneName, viewDrawComp, swControlSketch, viewComp, oSubAssy.StartComp, swView, swDrawing)
            Call AddSplitLineNote(swStartSketch, swDrawing, swView, "EXTERNAL WALL-" & Right(PlaneName, 1), False, 0.035)
        
        End If
        
        If Not InStr(oSubAssy.EndComp.GetCustomProperty("Profile"), "CORNER") > 0 Then
        
            PlaneName = GetPlaneName(wallName, False)
            Set swEndSketch = CreateSketchLinesForNonCornerPanels(PlaneName, viewDrawComp, swControlSketch, viewComp, oSubAssy.EndComp, swView, swDrawing)
            Call AddSplitLineNote(swEndSketch, swDrawing, swView, "EXTERNAL WALL-" & Right(PlaneName, 1))
            
        End If
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        
        If Not swStartSketch Is Nothing Then
        
            MaxClearance = MaxClearance + 0.008
            
            If Not swEndSketch Is Nothing Then
                
                swStartSketch.Select4 False, Nothing
                swEndSketch.Select4 True, Nothing

            Else
            
                swStartSketch.Select4 False, Nothing
                Call SelectEntity(oSubAssy.EndEdge, True, swView)
    
            End If
            
        Else

            If Not swEndSketch Is Nothing Then
            
                MaxClearance = MaxClearance + 0.008
                swEndSketch.Select4 False, Nothing
                Call SelectEntity(oSubAssy.StartEdge, True, swView)
            
            End If
            
        End If
        
        Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.StartComp.xMin + 0.01, oSubAssy.EndComp.yMin - MaxClearance, 0)
        If Not swDisplayDim Is Nothing Then

            swDisplayDim.CenterText = True
            swDisplayDim.SetDual2 False, False
            
            Dim AreaInSqft As Double
            AreaInSqft = (swDisplayDim.GetDimension2(0).Value * 0.0254 * swView.ScaleDecimal * oSubAssy.GetMaxLength) - oSubAssy.TotalDoorArea
            AreaInSqft = Round((AreaInSqft / (swView.ScaleDecimal * swView.ScaleDecimal)) * 10.7639, 2)
            
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "(" & AreaInSqft & " sq.ft)"
            
        End If

    End If
    

End Sub

Private Function CreateSketchLinesForNonCornerPanels(PlaneName As String, viewDrawComp As SldWorks.DrawingComponent, swControlSketch As SldWorks.Component2, _
                                        viewComp As SldWorks.Component2, oComp As IComp, swView As SldWorks.View, _
                                        swDrawing As SldWorks.ModelDoc2) As SldWorks.SketchSegment
                    
        Dim xMin As Double
        Dim yMin As Double
        Dim xMax As Double
        Dim yMax As Double
        Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax, yMax + 16 * 0.0254, _
                                    0, xMax, yMin - 16 * 0.0254, 0)
        swSketchSegment.ConstructionGeometry = True
            
        swDrawing.Extension.SelectByID2 PlaneName & "@" & viewDrawComp.Name & "@" & swView.Name & "/" & swControlSketch.Name & "@" & viewComp.Name2, "PLANE", 0, 0, 0, False, 0, Nothing, 0
        swSketchSegment.Select4 True, Nothing
            
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
        Set CreateSketchLinesForNonCornerPanels = swSketchSegment

End Function

Private Function GetPlaneName(wallName As String, IsLeftPanel As Boolean) As String

    If IsLeftPanel Then
        
        Select Case wallName
            
            Case "Wall-A"
            
                GetPlaneName = "Outside D"
            
            Case "Wall-B"
            
                GetPlaneName = "Outside A"
            
            Case "Wall-C"
            
                GetPlaneName = "Outside B"
            
            Case "Wall-D"
            
                GetPlaneName = "Outside C"

        End Select
        
    Else
        
        Select Case wallName
            
            Case "Wall-A"
            
                GetPlaneName = "Outside B"
            
            Case "Wall-B"
            
                GetPlaneName = "Outside C"
            
            Case "Wall-C"
            
                GetPlaneName = "Outside D"
            
            Case "Wall-D"
            
                GetPlaneName = "Outside A"

        End Select
        
    End If
           
End Function

Private Sub AddDimensionsForDoororHVACInEachSubAssy(subAssylist As IArrListObject, swDrawing As SldWorks.DrawingDoc, _
            swView As SldWorks.View, MaxClearance As Double, IsSectionNeeded As Boolean)

    Dim vSubAssy As Variant
    vSubAssy = subAssylist.Items

    MaxClearance = 0
    
    Dim i As Integer
    For i = LBound(vSubAssy) To UBound(vSubAssy)
    
        Dim oSubAssy As ISubAssy
        Set oSubAssy = vSubAssy(i)
        
        Dim Clearance As Double
        Clearance = 0.005
        
        If (UBound(vSubAssy)) = 0 Or i < UBound(vSubAssy) Then
            
            Call AddDimensionsForDoororHVAC(oSubAssy.GetDoorOrHVACAssemblies, oSubAssy, swDrawing, swView, Clearance, IsSectionNeeded)
            
        Else
        
            MaxClearance = MaxClearance + 0.008
            Call AddOverallDimension(oSubAssy, swDrawing, swView, MaxClearance)
            
        End If

        
        If Clearance > MaxClearance Then
        
            MaxClearance = Clearance
            
        End If
    
    Next i

End Sub

Private Sub AddOverallDimension(oSubAssy As ISubAssy, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, Clearance As Double)

    Dim swDisplayDim As SldWorks.DisplayDimension
    Set swDisplayDim = SelectAndAddDimension(oSubAssy.StartEdge, oSubAssy.EndEdge, swDrawing, _
                oSubAssy.EndComp.xMin - 0.01, oSubAssy.EndComp.yMin - Clearance, swView)
    Set oSubAssy.Dimension = swDisplayDim
    
End Sub

Private Sub AddDimensionsForDoororHVAC(vDoorOrHVACItems As Variant, oSubAssy As ISubAssy, swDrawing As SldWorks.DrawingDoc, _
                    swView As SldWorks.View, ByRef Clearance As Double, IsSectionNeeded As Boolean)

    Dim j As Integer
    
    If Not IsEmpty(vDoorOrHVACItems) Then
    
        Dim swDisplayDim As SldWorks.DisplayDimension
        
        For j = LBound(vDoorOrHVACItems) To UBound(vDoorOrHVACItems)
                
            Dim oDoorOrHVACAssy As IDoorOrHVACAssy
            Set oDoorOrHVACAssy = vDoorOrHVACItems(j)

            Dim oStartComp As IComp
            Set oStartComp = oDoorOrHVACAssy.StartComp
                
            Dim swDoorOrHVACStartEdge As SldWorks.Edge
            Set swDoorOrHVACStartEdge = GetEdgeInView(oStartComp, swView, False, True)

            Set swDisplayDim = SelectAndAddDimension(oSubAssy.StartEdge, swDoorOrHVACStartEdge, swDrawing, _
                        oStartComp.xMax - 0.001, oStartComp.yMin - Clearance, swView, False)
                        
            If oDoorOrHVACAssy.IsDoor Or False = IsSectionNeeded Then
            
                Dim oEndComp As IComp
                Set oEndComp = oDoorOrHVACAssy.EndComp
                
                Dim swDoorOrHVACEndEdge As SldWorks.Edge
                Set swDoorOrHVACEndEdge = GetEdgeInView(oEndComp, swView, False, False)
                
                Set swDisplayDim = SelectAndAddDimension(swDoorOrHVACStartEdge, swDoorOrHVACEndEdge, swDrawing, _
                        oEndComp.xMin - 0.001, oStartComp.yMin - Clearance, swView, False)
                        
'            Else
'
'                swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "TYP."
            
            End If
            
            Clearance = Clearance + 0.005
                
        Next j
            
    End If
    
    Clearance = Clearance + 0.004
    Call AddOverallDimension(oSubAssy, swDrawing, swView, Clearance)

End Sub

Private Sub CheckAndAddDoorOrHVACAssy(subAssylist As IArrListObject, DoorOrHVACList As IArrListObject, CompNoDict As Scripting.Dictionary)
    
    Dim vSubAssemblies As Variant
    vSubAssemblies = subAssylist.Items
    
    Dim vDoorOrHVACItems As Variant
    vDoorOrHVACItems = DoorOrHVACList.Items
    
    Dim i As Integer
    Dim j As Integer
    
    Dim LastSubAssyIdx As Integer
    LastSubAssyIdx = 0
    
    If Not IsEmpty(vDoorOrHVACItems) Then
    
    For i = LBound(vDoorOrHVACItems) To UBound(vDoorOrHVACItems)
    
        Dim oDoorOrHVACAssy As IDoorOrHVACAssy
        Set oDoorOrHVACAssy = vDoorOrHVACItems(i)
        
        Dim AssyIdx As Integer
        AssyIdx = CompNoDict.Item(oDoorOrHVACAssy.EndComp.GetComponent.Name2)
        
        For j = LastSubAssyIdx To UBound(vSubAssemblies)
        
            Dim oSubAssy As ISubAssy
            Set oSubAssy = vSubAssemblies(j)
            
            If AssyIdx <= oSubAssy.EndIdx Then
                
                Call oSubAssy.AddDoororHVACAssy(oDoorOrHVACAssy)
                LastSubAssyIdx = j
                Exit For
                
            End If
            
       Next j

    Next i
    
    End If
    
End Sub

Private Sub AddDimensionNames(subAssylist As IArrListObject, wallName As String, swView As SldWorks.View)

        Dim CloneList As IArrListObject
        Set CloneList = New IArrListObject
        
        Set CloneList = subAssylist.Clone
        
        Dim isRoof As Boolean
        isRoof = False
        
        If InStr(wallName, "Wall") > 0 Then
        
            CloneList.SortItems "AssyLength"
        
        End If
        
        If InStr(UCase(wallName), "ROOF") > 0 Then
        
            isRoof = True
            
        End If
    
        
        Dim i As Integer
        Dim vSubAssy As Variant
        vSubAssy = CloneList.Items
        
        For i = LBound(vSubAssy) To UBound(vSubAssy)
        
            Dim oSubAssy As ISubAssy
            Set oSubAssy = vSubAssy(i)
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            Set swDisplayDim = oSubAssy.Dimension
            
            If Not swDisplayDim Is Nothing Then
            
                Dim AssyName As String
                If InStr(wallName, "-") Then
                
                    AssyName = UCase(wallName) & i + 1
                    
                Else
                    
                    AssyName = UCase(wallName) & "-" & i + 1
                    
                End If
                
                If isRoof Then
                
                    If Not i = UBound(vSubAssy) Then
                    
                        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, AssyName
                        
                    End If
                
                Else
                
                    Dim AreaInSqft As Double
                    AreaInSqft = ((oSubAssy.EndComp.xMax - oSubAssy.StartComp.xMin) * oSubAssy.GetMaxLength) - oSubAssy.TotalDoorArea
                    AreaInSqft = Round((AreaInSqft / (swView.ScaleDecimal * swView.ScaleDecimal)) * 10.7639, 2)
                    
                    If i = UBound(vSubAssy) Then
                        
                        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "(" & AreaInSqft & " sq.ft)"
                    
                    Else
                    
                        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, AssyName & vbCrLf & "(" & AreaInSqft & " sq.ft)"
                        
                    End If

                    
                End If
                
            End If
        
        Next i
        

    
End Sub

Private Function GetSubAssyComponentsIndexSorted(vComps As Variant, CompNoDict As Scripting.Dictionary)
    
    Dim vIdxArr As IArrList
    Set vIdxArr = New IArrList
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        vIdxArr.AddtoList CompNoDict(vComps(i).Name2)
        
    Next i
    
    vIdxArr.SortItems False
    GetSubAssyComponentsIndexSorted = vIdxArr.Items

End Function

Private Function AddSplitLines(vCompsIdx As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
        compDict As Scripting.Dictionary, CompNoDict As Scripting.Dictionary, IsFrontView As Boolean, _
        ByVal swLeftEdge As SldWorks.Edge, ByVal swRightEdge As SldWorks.Edge, Optional VisibleEdgesOnly As Boolean = True) As IArrListObject

    swDrawing.ActivateView swView.Name
    
    Dim vOutline As Variant
    vOutline = swView.GetOutline
    
    Dim i As Integer
    Dim NextAssyStartEdge As SldWorks.Edge
    
    Dim subAssylist As IArrListObject
    Set subAssylist = New IArrListObject
    
    Dim swLayerMgr As SldWorks.LayerMgr
    Set swLayerMgr = swDrawing.GetLayerManager
    
    Const LayName As String = "SPLIT LINE"
    
    Call CheckandAddLayer(LayName, "ASSEMBLY SPLIT LINE", swLayerMgr)

    For i = LBound(vCompsIdx) To UBound(vCompsIdx)
    
        Dim xMin As Double
        Dim yMin As Double
        Dim xMax As Double
        Dim yMax As Double

        Dim oComp As IComp
        Set oComp = compDict.Items(vCompsIdx(i))
            
        Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax, yMax + 16 * 0.0254, _
                                0, xMax, yMin - 16 * 0.0254, 0)
                                
        'swSketchSegment.ConstructionGeometry = True
        swSketchSegment.Layer = LayName
        
        Call AddSplitLineNote(swSketchSegment, swDrawing, swView, "SPLIT LINE", False, 0.02)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp, swView, False, True, VisibleEdgesOnly)
        
        Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)
        
        Dim oSubAssy As ISubAssy
        Dim swDisplayDim As SldWorks.DisplayDimension
    
        If IsFrontView Then
            
            If i = LBound(vCompsIdx) Then
            
                Set oSubAssy = New ISubAssy
                'Set swDisplayDim = SelectAndAddDimension(swLeftEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vOutline(1) - 0.015, swView)
                             
                Set oSubAssy.StartComp = compDict.Items(0)
                Set oSubAssy.EndComp = compDict.Items(vCompsIdx(i))
                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                'Set oSubAssy.Dimension = swDisplayDim
                'oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = 0
                oSubAssy.EndIdx = vCompsIdx(i)
                
                subAssylist.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                'Set swDisplayDim = SelectAndAddDimension(NextAssyStartEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vOutline(1) - 0.015, swView)
                                
                Set oSubAssy.StartComp = compDict.Items(vCompsIdx(i - 1) + 1)
                Set oSubAssy.EndComp = compDict.Items(vCompsIdx(i))
                                
                Set oSubAssy.StartEdge = NextAssyStartEdge
                Set oSubAssy.EndEdge = swEdge
                'Set oSubAssy.Dimension = swDisplayDim
                'oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = vCompsIdx(i - 1) + 1
                oSubAssy.EndIdx = vCompsIdx(i)
                
                subAssylist.AddtoList oSubAssy

            End If
            
            Dim NextAssyComp As IComp
            Set NextAssyComp = compDict.Items(vCompsIdx(i) + 1)
            
            Set NextAssyStartEdge = GetEdgeInView(NextAssyComp, swView, False, False, False)
            
            If i = UBound(vCompsIdx) Then
            
                Set oSubAssy = New ISubAssy
                'Set swDisplayDim = SelectAndAddDimension(swRightEdge, NextAssyStartEdge, swDrawing, _
                            NextAssyComp.xMax + 0.01, vOutline(1) - 0.015, swView)
                            
                Set oSubAssy.StartComp = compDict.Items(vCompsIdx(i) + 1)
                Set oSubAssy.EndComp = compDict.Items(UBound(compDict.Items))
                
                Set oSubAssy.StartEdge = NextAssyStartEdge
                Set oSubAssy.EndEdge = swRightEdge
                'Set oSubAssy.Dimension = swDisplayDim
                'oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                
                oSubAssy.StartIdx = vCompsIdx(i) + 1
                oSubAssy.EndIdx = (CompNoDict.Count) - 1
                
                subAssylist.AddtoList oSubAssy
                            
            End If
            
        Else
            
            Dim TempComp As IComp
            Dim vPoint(2) As Double
            Dim vSheetPoint As Variant
    
            If i = LBound(vCompsIdx) Then
                
                Set TempComp = compDict.Items(0)
                Call GetViewMaxMinPoints(TempComp, swView, xMin, xMax, yMin, yMax)
                vPoint(0) = xMin
                vPoint(1) = yMin
                vPoint(2) = 0
                
                vSheetPoint = GetSketchPointInSheetSpace(swView, vPoint)
                
                Set swLeftEdge = GetEdgeInView(TempComp, swView, False, False)
                
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swLeftEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView)
                                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(subAssylist.Items(UBound(subAssylist.Items)).EndEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView)
                                
                Set oSubAssy.StartEdge = subAssylist.Items(UBound(subAssylist.Items)).EndEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy

            End If
            
            
            If i = UBound(vCompsIdx) Then
            
                Set TempComp = compDict.Items(UBound(compDict.Items))
                Set swRightEdge = GetEdgeInView(TempComp, swView, False, True)
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swEdge, swRightEdge, swDrawing, _
                            oComp.xMax + 0.01, vSheetPoint(1) - 0.005, swView)
                            
                Set oSubAssy.StartEdge = swEdge
                Set oSubAssy.EndEdge = swRightEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy
                            
            End If
            
        End If
        
    Next i
    
    Set AddSplitLines = subAssylist
    
End Function

Private Function GetControlSketch() As SldWorks.Component2

    Dim swTopLevelAssy As SldWorks.AssemblyDoc
    Set swTopLevelAssy = swTopLevelModel
    
    Dim vComps As Variant
    vComps = swTopLevelAssy.GetComponents(True)
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)
        
        If InStr(swComp.Name2, "CONTROL") > 0 And InStr(swComp.Name2, "SKETCH") > 0 Then
            
            Dim vBodies As Variant
            Dim vBodiesInfo As Variant
            vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodiesInfo)
            
            If IsEmpty(vBodies) Then
            
                Set GetControlSketch = swComp
                Exit Function
                
            End If
                
            
        End If

    Next i
       
End Function

Private Sub AddSplitLineNote(swSketchSegment As SldWorks.sketchLine, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            NoteText As String, Optional IsRight As Boolean = True, Optional ClearanceVal As Double = 0.005)

    
    Dim vPointInSheet As Variant
    
'    If InStr(NoteText, "SPLIT") > 0 Then
    
'        vPointInSheet = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, False, 0)
'
'    Else
'
        vPointInSheet = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, False)
        
'    End If
    
    If IsRight Then
    
        Call AddNoteToView(swDrawing, NoteText, vPointInSheet(0) + ClearanceVal, vPointInSheet(1) + 0.00625)
        
    Else
        
        Call AddNoteToView(swDrawing, NoteText, vPointInSheet(0) - ClearanceVal, vPointInSheet(1) + 0.00625)
        
    End If
    
End Sub

Private Function GetCompDictionary(FlatCompList As Variant, CompNoDict As Scripting.Dictionary) As Scripting.Dictionary
    
    Dim TempDict As New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(FlatCompList) To UBound(FlatCompList)
        
        TempDict.Add FlatCompList(i).GetComponent.Name2, FlatCompList(i)
        CompNoDict.Add FlatCompList(i).GetComponent.Name2, i
    
    Next i
    
    Set GetCompDictionary = TempDict
    
End Function

Function GetSelectedComponents() As Variant
    
    Dim swSelectionMgr As SldWorks.SelectionMgr
    Set swSelectionMgr = swTopLevelModel.SelectionManager
    
    Dim compDict As Scripting.Dictionary
    Set compDict = New Scripting.Dictionary
    
    If swSelectionMgr.GetSelectedObjectCount2(-1) > 0 Then
    
        Dim i As Integer
        For i = 0 To swSelectionMgr.GetSelectedObjectCount2(-1) - 1
            
            Dim swComp As SldWorks.Component2
            Set swComp = swSelectionMgr.GetSelectedObjectsComponent4(i + 1, -1)
            
            If False = compDict.Exists(swComp.Name2) Then
                
                compDict.Add swComp.Name2, swComp
            
            End If

        Next i
        
    End If
    
    If Not (compDict.Count = 0) Then
    
        GetSelectedComponents = compDict.Items
        
    End If

End Function
Private Sub ActivateDrawingDocument(swModel As SldWorks.ModelDoc2)
    
    Dim swFrame As SldWorks.Frame
    Set swFrame = swApp.Frame
    
    Dim vModelWindows As Variant
    vModelWindows = swFrame.ModelWindows
    
    Dim i As Integer
    For i = LBound(vModelWindows) To UBound(vModelWindows)
    
        Dim swModelWindow As SldWorks.ModelWindow
        Set swModelWindow = vModelWindows(i)
        
        If swModelWindow.Title = swModel.GetTitle Then
        
            swModelWindow.Activate
            Exit Sub
            
        End If
    
    Next i
End Sub

Private Function CheckForMultipleAssembly(ViewWidth As Double, ViewHeight As Double) As Boolean

    CheckForMultipleAssembly = False
    
    If ViewHeight <= 165 * 0.0254 Then
    
        If ViewWidth > 450 * 0.0254 Then
    
            CheckForMultipleAssembly = True
            
        End If
        
    Else
        
        If ViewWidth > 165 * 0.0254 Then
            
            CheckForMultipleAssembly = True
            
        End If
        
    End If

End Function

Private Sub AddLegendBlocks(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet)

    swDrawing.ActivateSheet swSheet.GetName
    
    Dim vSheetProp As Variant
    vSheetProp = swSheet.GetProperties
    
    Dim vPt(2) As Double
    vPt(0) = 0.1025 * vSheetProp(3)
    vPt(1) = 0.0161 * vSheetProp(3)
    vPt(2) = 0
    
    Dim SketchBlockInsertionPt As SldWorks.MathPoint
    Set SketchBlockInsertionPt = swMathUtility.CreatePoint(vPt)
    
    Dim swBlockDefinition As SldWorks.SketchBlockDefinition
    
'    If IsZChannelExists Then
    
         Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(SketchBlockInsertionPt, _
                "C:\FBD\COMMON\BLOCKS\MAKE UP+ASSEMBLY+ PART NUMBER LEGEND.SLDBLK", False, 0.02, 0)

'    Else
'
'        Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(Nothing, _
'                "C:\FBD\COMMON\BLOCKS\MAKEUP PANEL LEGEND FOR 6 SERIES.SLDBLK", False, vSheetProp(2) / vSheetProp(3), 0)
'
'    End If
    

    'Set swBlockInst = swDrawing.SketchManager.InsertSketchBlockInstance(swBlockDefinition, SketchBlockInsertionPt, 1, 0)

    
    swDrawing.GraphicsRedraw2

End Sub

Private Function AddDimensionInFrontView(swView As SldWorks.View, FlatCompList As Variant, _
            DetailedCompList As Variant, MaxCompHeight As IComp, swDrawing As SldWorks.ModelDoc2, _
            ByRef swLeftEdge As SldWorks.Edge, ByRef swRightEdge As SldWorks.Edge) As SldWorks.Edge
            
    Dim vOutline As Variant
    vOutline = swView.GetOutline

    Dim LeftComp As IComp
    Set LeftComp = FlatCompList(0)
    
    Dim RightComp As IComp
    Set RightComp = FlatCompList(UBound(FlatCompList))
    
    Set swLeftEdge = GetEdgeInView(LeftComp, swView, False, False)
    Set swRightEdge = GetEdgeInView(RightComp, swView, False, True)
            
    Dim swTopRightEdge As SldWorks.Edge
    Set swTopRightEdge = GetEdgeInView(RightComp, swView, True, True)
    
    Dim swBottomRightEdge As SldWorks.Edge
    Set swBottomRightEdge = GetEdgeInView(RightComp, swView, True, False)
    
    Dim swRightDim As SldWorks.DisplayDimension

    If (Abs(LeftComp.yMax - RightComp.yMax) <= 0.5 * 0.0254 * swView.ScaleDecimal) Then
    
        Dim MaxCompEdge As SldWorks.Edge
        Set MaxCompEdge = GetEdgeInView(MaxCompHeight, swView, True, True)
        
        Set swRightDim = SelectAndAddDimension(MaxCompEdge, _
                        swBottomRightEdge, swDrawing, RightComp.xMax + 0.025, (vOutline(1) + vOutline(3)) / 2, swView)
    Else
    
        Dim swBottomLeftEdge As SldWorks.Edge
        Set swBottomLeftEdge = GetEdgeInView(LeftComp, swView, True, False)
        
        Dim swTopLeftEdge As SldWorks.Edge
        Set swTopLeftEdge = GetEdgeInView(LeftComp, swView, True, True)
        
        Set swRightDim = SelectAndAddDimension(swTopRightEdge, _
                        swBottomRightEdge, swDrawing, RightComp.xMax + 0.025, (vOutline(1) + vOutline(3)) / 2, swView)
                        
        Dim swLeftDim As SldWorks.DisplayDimension
        Set swLeftDim = SelectAndAddDimension(swTopLeftEdge, _
            swBottomLeftEdge, swDrawing, LeftComp.xMin - 0.015, (vOutline(1) + vOutline(3)) / 2, swView)
        
    End If
    
    Set AddDimensionInFrontView = swBottomRightEdge

End Function

Private Function SelectAndAddDimension(swEdge1 As SldWorks.Edge, swEdge2 As SldWorks.Edge, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, YPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True) As SldWorks.DisplayDimension
    
    If Not (swEdge1 Is Nothing) And Not (swEdge2 Is Nothing) Then
        
        swDrawing.ClearSelection2 True
        Call SelectEntity(swEdge1, False, swView)
        Call SelectEntity(swEdge2, True, swView)
        
        Set SelectAndAddDimension = swDrawing.AddHorizontalDimension2(xPos, YPos, 0)
        
        If Not SelectAndAddDimension Is Nothing Then
        
            SelectAndAddDimension.CenterText = True
            
            If IsDual Then
            
                SelectAndAddDimension.SetDual2 False, False
                
            End If
            
        End If
    
    End If

End Function

Private Sub GetViewMaxMinPoints(oComp As IComp, swView As SldWorks.View, ByRef xMin As Double, _
                ByRef xMax As Double, ByRef yMin As Double, ByRef yMax As Double)

    Dim vViewMaxPt As Variant
    vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
    Dim vViewMinPt As Variant
    vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(0), vViewMaxPt(0), xMin, xMax)
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(1), vViewMaxPt(1), yMin, yMax)
    
End Sub
 
Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, Is12GAPanelExists As Boolean, _
            IsAllPanels12GA As Boolean, IsDoorExists As Boolean, ByRef NoteCount As Integer, _
                wallName As String, AssyCutCount As Integer) As SldWorks.Note

    swDrawing.ActivateSheet swSheet.GetName
    
    Dim swStructuralNote As SldWorks.Note
    Dim Note As String
    
    If Is12GAPanelExists Then
    
        NoteCount = 2
        If IsAllPanels12GA Then
        
            Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
                "<FONT size=8PTS style=R>1. ALL PANELS ARE 12GA." & vbCrLf & _
             "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
        
        Else
            Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
                "<FONT size=8PTS style=R>1. ALL CIRCLED PANELS ARE 12GA." & vbCrLf & _
             "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
             
        End If

    Else
    
        NoteCount = 1
        Note = "<FONT size=10PTS style=B> NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
         
     End If
     
    
    If AssyCutCount > 0 Then
        
        NoteCount = NoteCount + 1
        Note = Note & vbCrLf & NoteCount & ". VERIFY THE POSITION OF OEM BLOCKOUT WITH RESPECT TO L-TABS IN LINER PANEL."
        
    End If
     
    
    If InStr(wallName, "Wall") > 0 Then

        If IsDoorExists Then
     
            NoteCount = NoteCount + 1
            Note = Note & vbCrLf & NoteCount & ". DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM HORIZONTAL FACE OF DOOR C-CHANNEL."
        
        End If
        
        NoteCount = NoteCount + 1
        Note = Note & vbCrLf & NoteCount & ". DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM OF CEILING PANELS, USE FOR CEILING L-ANGLE PLACEMENT."
        
    End If
     
    Set swStructuralNote = swDrawing.CreateText2(Note, 1.99241243641486E-02, 6.92464210842187E-02, 0, 0, 0)
    swStructuralNote.SetTextJustification swTextJustification_e.swTextJustificationLeft
    
End Function

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
                "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " INTERNAL ELEVATION KEY.SLDBLK", True, 1, 0)
                

End Sub

Private Sub UpdateBottomViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim oComp As IComp
    Set oComp = vComps(0)
    
    Dim viewPosition As Variant
    viewPosition = swView.Position
    
    Dim wallPanelPosition As Variant
    wallPanelPosition = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
    
    If Abs(wallPanelPosition(1) - viewPosition(1)) / swView.ScaleDecimal > 5 * 0.0254 Then
    
        viewPosition(1) = viewPosition(1) + viewPosition(1) - wallPanelPosition(1)
        
    End If
    
    swView.Position = viewPosition
    
End Sub

Private Sub CleanUpActivateAndAddViewLabel(swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, wallName As String, YPos As Double, _
    Optional xPos As Double = 0)

    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayOrigins, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayPlanes, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints2, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCurves, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayAllAnnotations, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCompAnnotations, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLiveSections, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLights, False
    
    swDrawing.ActivateView swView.Name
    
    Dim SheetDesc As String
    Dim LabelText As String
    If InStr(wallName, "Wall") > 0 Then
    
        SheetDesc = "STRUCTURAL, ELEVATION, EXTERNAL PANELS, " & UCase(wallName)
        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34) & _
         vbCrLf & "<FONT size=8PTS style=R> (INTERIOR VIEW)"
         
    Else
    
        SheetDesc = "STRUCTURAL, " & UCase(wallName)
        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34)
        
    End If
    swDrawing.Extension.CustomPropertyManager("").Set2 "SHEET DESCRIPTION", SheetDesc
    swDrawing.Extension.CustomPropertyManager("").Set2 "ISSUED FOR", "CONSTRUCTION"
    
    If xPos = 0 Then
    
        Dim vOutline As Variant
        vOutline = swView.GetOutline
        xPos = (vOutline(0) + vOutline(2)) / 2
        
    End If
    
    Dim swLabelNote As SldWorks.Note

    Set swLabelNote = swDrawing.CreateText2(LabelText, xPos, YPos, 0, 0, 0)
    swLabelNote.SetTextJustification swTextJustification_e.swTextJustificationCenter
    
    swDrawing.Extension.Rebuild swRebuildOptions_e.swCurrentSheetDisp

End Sub

Private Function Add12GACircles(vCompList As Variant, swDrawing As SldWorks.ModelDoc2, _
                swView As SldWorks.View, wallName As String, ByRef IsAllPanels12GA As Boolean) As Boolean
    
    Add12GACircles = False
    IsAllPanels12GA = True
    
    swDrawing.ActivateView swView.Name
    'swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
    
    Dim i As Integer
    For i = LBound(vCompList) To UBound(vCompList)
    
        Dim oComp As IComp
        Set oComp = vCompList(i)

        Dim vViewMaxPt As Variant
        vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
        Dim vViewMinPt As Variant
        vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
            
        If oComp.GetCustomProperty("THK") = 0.1084 Then
        
            Add12GACircles = True
            
            If InStr(wallName, "Wall") > 0 Then
            
                Dim vPt(2) As Double
                vPt(0) = (vViewMaxPt(0) + vViewMinPt(0)) / 2
                vPt(1) = (vViewMaxPt(1) + vViewMinPt(1)) / 2
                vPt(2) = (vViewMaxPt(2) + vViewMinPt(2)) / 2
                     
                Dim radius As Double
                radius = Sqr((vViewMaxPt(0) - vPt(0)) ^ 2 + (vViewMaxPt(1) - vPt(1)) ^ 2) + 0.0127
            
                Dim swSketchSegment As SldWorks.SketchSegment
                Set swSketchSegment = swSketchMgr.CreateCircleByRadius(vPt(0), vPt(1), vPt(2), radius)
                swSketchSegment.ConstructionGeometry = True
                
            End If
            
        Else
        
            IsAllPanels12GA = False
                
        End If
            
        If i = UBound(vCompList) Or i = LBound(vCompList) Then
                
            Call AddRibSketchAndNote(oComp, swView, swSketchMgr, swDrawing, i)
            
        End If

    Next i
    
    
    'swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
        
End Function

Private Sub AddRibSketchAndNote(oComp As IComp, swView As SldWorks.View, swSketchMgr As SldWorks.SketchManager, _
                swDrawing As SldWorks.DrawingDoc, CompPos As Integer)
    
    Dim Profile As String
    Profile = oComp.GetCustomProperty("Profile")
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
        
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
    
    Dim vSketchPoint As Variant
    
    If InStr(Profile, "CORNER") > 0 Then
    
        Dim swSketchSegmentHor As SldWorks.SketchSegment
        Dim swSketchSegmentVer As SldWorks.SketchSegment
    
        If InStr(Profile, "1") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 3.5, 1.5)
         
        ElseIf InStr(Profile, "2") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 1.5, 1.5)

        
        ElseIf InStr(Profile, "3") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 3.5, 3.5)
                
        
        ElseIf InStr(Profile, "4") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 1.5, 3.5)
        
        End If
        
        If Not swSketchSegmentHor Is Nothing And Not swSketchSegmentVer Is Nothing Then
    
            Dim bool As Boolean
            bool = swDrawing.ActivateView(swView.Name)
            
            Call SelectSketchSegment(swSketchSegmentHor, swDrawing, swView, False)
            

            vSketchPoint = SelectSketchSegment(swSketchSegmentVer, swDrawing, swView, True)
            
            Call AddNoteToView(swDrawing, "RIB TO RIB" & vbCrLf & "#14 TEK SCREW" & vbCrLf & "@ 6" & Chr(34) & " O.C.", _
                            vSketchPoint(0) + 0.0075, vSketchPoint(1) + 0.0125)

        End If
    
    End If
    
    If Not (CompPos = 0) Then
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax - 0.25 * 0.0254, yMin, _
                                0, xMax + 16 * 0.0254, yMin, 0)
                                
        swSketchSegment.ConstructionGeometry = True
        
        vSketchPoint = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, True, 0.5)
        Call AddNoteToView(swDrawing, "CASTING BED", vSketchPoint(0) + 0.0075, vSketchPoint(1) - 0.005)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp, swView, True, False)
        
        Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)

    End If
    
End Sub

Private Sub AddCollinearRelation(swDrawing As SldWorks.DrawingDoc, swEdge As SldWorks.Edge, swSketchSegment As SldWorks.SketchSegment, swView As SldWorks.View)
    
    If Not (swEdge Is Nothing) And Not (swSketchSegment Is Nothing) Then
    
        Call SelectEntity(swEdge, False, swView)
        swSketchSegment.Select4 True, Nothing
                
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
    End If
    
End Sub

Function GetEdgeInView(oComp As IComp, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, _
    Optional CheckAllVisibleEdgesOnly As Boolean = True, Optional IsSection As Boolean = False) As SldWorks.Edge
    
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
    
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
    
    Dim swComp As SldWorks.Component2
    Set swComp = oComp.GetComponent

    Dim TempLength As Double

    Dim vEnts As Variant
    Dim vPolyLinesBuffer As Variant
    
    If IsSection Then
    
        vEnts = swView.GetPolylines7(1, vPolyLinesBuffer)
    
    Else
        If CheckAllVisibleEdgesOnly Then
        
            vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
            
        Else
        
            vEnts = GetComponentEdges(swComp)
            
        End If
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)
            
            Dim swEntity As SldWorks.Entity
            Set swEntity = swEdge
                    
            Dim swEntityComp As SldWorks.Component2
            Set swEntityComp = swEntity.GetComponent
            
            
            Dim IsSelected As Boolean
            'IsSelected = SelectEntity(swEdge, False, swView)
            
            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
                If swCurve.IsLine Then
                
                    Dim vStartPoint As Variant
                    vStartPoint = swEdge.GetStartVertex.GetPoint
                    vStartPoint = GetComponentPointInViewSpace(swComp, vStartPoint, swView)
                    
                    Dim vEndPoint As Variant
                    vEndPoint = swEdge.GetEndVertex.GetPoint
                    vEndPoint = GetComponentPointInViewSpace(swComp, vEndPoint, swView)
                    
                    If Abs(vStartPoint(idx) - vEndPoint(idx)) <= 0.00001 And Abs(vStartPoint(idx) - ValToMatch) <= 0.00001 And _
                                Abs(vStartPoint(2) - vEndPoint(2)) <= 0.000001 Then
                        
                        Dim vCurveParam As Variant
                        vCurveParam = swEdge.GetCurveParams2
    
                        If IsSection Then
                            
                            If Not InStr(swComp.Name2, swEntityComp.Name2) > 0 Then
                                
                                GoTo NextIteration
                                
                            End If
                            
                        End If
                        
                        If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
    
                            TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                            Set GetEdgeInView = swEdge
    
                        End If
                        
                    End If
                
                End If
                
NextIteration:
            
        Next i

    End If

End Function


Sub AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, YPos As Double)
            
    Dim swNote As SldWorks.Note
    Set swNote = swDrawing.InsertNote(NoteText)
            
    If Not swNote Is Nothing Then

        Dim swAnnotation As SldWorks.Annotation
        Set swAnnotation = swNote.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, YPos, 0

        End If

    End If
    
End Sub

Sub CreateRibSketches(ByRef swSketchSegmentHor As SldWorks.SketchSegment, ByRef swSketchSegmentVer As SldWorks.SketchSegment, _
                    xMin As Double, xMax As Double, yMin As Double, yMax As Double, CompPos As Integer, swSketchMgr As SldWorks.SketchManager, _
                        OffSetVer As Double, OffSetHor As Double)
                    
    Const Length As Double = 3
    Const FrontOffset As Double = 1.5

    If CompPos = 0 Then
    
        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMax - OffSetHor * 0.0254, yMin + FrontOffset * 0.0254, _
                                0, xMax - (OffSetHor - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)
                                
        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMin + FrontOffset * 0.0254, yMax - OffSetVer * 0.0254, _
                                0, xMin + FrontOffset * 0.0254, yMax - (OffSetVer - Length) * 0.0254, 0)
                                
    Else

        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMin + OffSetVer * 0.0254, yMin + FrontOffset * 0.0254, _
                                0, xMin + (OffSetVer - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)

        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMax - FrontOffset * 0.0254, yMax - OffSetHor * 0.0254, _
                                0, xMax - FrontOffset * 0.0254, yMax - (OffSetHor - Length) * 0.0254, 0)

    End If
    
End Sub

Function SelectSketchSegment(swSketchSegment As SldWorks.SketchSegment, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, Append As Boolean, Optional IsNearEnd As Boolean = True, Optional PercentFromEnd As Double = 0.01)
    
    Dim swSketchLine As SldWorks.sketchLine
    Set swSketchLine = swSketchSegment
    
    Dim swStartPoint As SldWorks.sketchPoint
    Set swStartPoint = swSketchLine.GetStartPoint2
    
    Dim swEndPoint As SldWorks.sketchPoint
    Set swEndPoint = swSketchLine.GetEndPoint2

    Dim swCurve As SldWorks.Curve
    Set swCurve = swSketchSegment.GetCurve
        
    Dim LineLength As Double
    LineLength = swSketchSegment.GetLength
            
    Dim vLineParams As Variant
    vLineParams = swCurve.LineParams
        
    Dim vVectorData(2) As Double
    vVectorData(0) = vLineParams(3)
    vVectorData(1) = vLineParams(4)
    vVectorData(2) = vLineParams(5)
        
    Dim swMathVector As SldWorks.MathVector
    Set swMathVector = swMathUtility.CreateVector(vVectorData)
        
    Set swMathVector = swMathVector.Normalise
        
    Dim vSketchPoint(2) As Double
    Dim swMathPoint As SldWorks.MathPoint
        
    If IsNearEnd Then
            
        vSketchPoint(0) = swEndPoint.X
        vSketchPoint(1) = swEndPoint.Y
        vSketchPoint(2) = swEndPoint.Z
            
        Set swMathPoint = swMathUtility.CreatePoint(vSketchPoint)
        Set swMathVector = swMathVector.Scale(-1 * PercentFromEnd * LineLength)
            
    Else
 
        vSketchPoint(0) = swStartPoint.X
        vSketchPoint(1) = swStartPoint.Y
        vSketchPoint(2) = swStartPoint.Z

        Set swMathVector = swMathVector.Scale(PercentFromEnd * LineLength)
            
    End If
        
    Set swMathPoint = swMathUtility.CreatePoint(vSketchPoint)
    Set swMathPoint = swMathPoint.AddVector(swMathVector)
        
    vSketchPoint(0) = swMathPoint.ArrayData(0)
    vSketchPoint(1) = swMathPoint.ArrayData(1)
    vSketchPoint(2) = swMathPoint.ArrayData(2)
 
    Dim vPointInSheet As Variant
    vPointInSheet = GetSketchPointInSheetSpace(swView, vSketchPoint)
    
    swDrawing.Extension.SelectByID2 "Line" & swSketchSegment.GetID(1), "SKETCHSEGMENT", vPointInSheet(0), vPointInSheet(1), vPointInSheet(2), Append, -1, Nothing, 0
    SelectSketchSegment = vPointInSheet
    
End Function



Sub UpdateHatchProperties(swView As SldWorks.View)
    
    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    Dim vSketchHatches As Variant
    vSketchHatches = swSketch.GetSketchHatches
            
    If Not IsEmpty(vSketchHatches) Then
            
        Dim i As Integer
        For i = LBound(vSketchHatches) To UBound(vSketchHatches)
                
            Dim swSketchHatch As SldWorks.SketchHatch
            Set swSketchHatch = vSketchHatches(i)
                    
            swSketchHatch.Pattern = "ANSI32 (Steel)"
            swSketchHatch.Scale2 = swView.ScaleDecimal * 3
                
        Next i
                
    End If

End Sub

Private Sub AddCallouts(vConsolidatedList As Variant, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
        MaxCompHeight As Double, ByRef IsMakeUpExists As Boolean, subAssyCompDict As Scripting.Dictionary)
    

    Const Increment As Double = 0.005
    Const MaxBalloonWidth As Double = 0.015875
    
    IsMakeUpExists = False
    
    swDrawing.Extension.SetUserPreferenceInteger swUserPreferenceIntegerValue_e.swDetailingBOMUpperText, swUserPreferenceOption_e.swDetailingNoOptionSpecified, swBalloonTextContent_e.swBalloonTextPartNumberBOM
    
    Dim maxNoOfBalloons As Integer
    maxNoOfBalloons = Int((SheetPosForLastBalloon - MaxCompHeight) / Increment)
    
    Dim AddorSub As Integer
    Dim BalloonCount As Integer
    
    AddorSub = -1
    BalloonCount = maxNoOfBalloons
     
    If InStr(vConsolidatedList(0).Comp.GetCustomProperty("Profile"), "CORNER") > 0 Then
     
        AddorSub = 1
        BalloonCount = 1
        
    End If
    
    Dim AnnXPos As Double
    Dim AnnYPos As Double
    
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
    
                    If Abs(prevComp.xMin - oComp.xMin) > 2 * MaxBalloonWidth Or _
                        Abs(prevComp.xMin - oComp.xMin) > MaxBalloonWidth And BalloonCount > 2 Then
    
                        AddorSub = 1
                        BalloonCount = 1
                        
    
                    End If
    
                Else
    
                    If Abs(prevComp.xMin - oComp.xMin) > MaxBalloonWidth Then
    
                        AddorSub = 1
                        BalloonCount = 1
    
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


Function InsertBalloonAndGetAnnotations(swDrawing As SldWorks.DrawingDoc, Qty As Integer, AnnXPos As Double, AnnYPos As Double, Optional BalloonStyle As swBalloonStyle_e = swBalloonStyle_e.swBS_Inspection) As SldWorks.Annotation
        
    Dim swBalloonParams As SldWorks.BalloonOptions
    Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
    swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
    swBalloonParams.Style = BalloonStyle
           
    If Qty > 1 Then
    
        swBalloonParams.ShowQuantity = True
        swBalloonParams.QuantityOverride = True
        swBalloonParams.QuantityOverrideValue = CStr(Qty)
                
    End If
    
    Dim swNote As SldWorks.Note
    Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
            
    If Not swNote Is Nothing Then

        Set InsertBalloonAndGetAnnotations = swNote.GetAnnotation
        InsertBalloonAndGetAnnotations.SetPosition2 AnnXPos, AnnYPos, 0
        
    End If
    
End Function

Private Sub AddHatchForMakeUpPanel(oComp As IComp, swDrawing As SldWorks.ModelDoc2, ByRef IsMakeUpExists As Boolean, swView As SldWorks.View)
    
    swDrawing.ClearSelection2 True
    
    If oComp.IsMakeUpPanel Then
    
        IsMakeUpExists = True
            
        Dim vFaces As Variant
        vFaces = oComp.GetViewNormalFaces
        
        Dim i As Integer
        For i = LBound(vFaces) To UBound(vFaces)
            
            Dim swFace As SldWorks.Face2
            Set swFace = vFaces(i)
            Call SelectEntity(swFace, True, swView)
            swDrawing.InsertHatchedFace
            
        Next i
            
    End If

End Sub

Private Sub SelectComponent(swDrawing As SldWorks.ModelDoc2, oComp As IComp, xPos As Double, _
    YPos As Double, Count As Integer, IsSelected As Boolean, swView As SldWorks.View)
    
    IsSelected = swDrawing.Extension.SelectByID2("", "FACE", xPos, YPos, _
                    0, False, -1, Nothing, 1)
                    
    If Count > 2 Then
        
        Dim vFaces As Variant
        vFaces = oComp.GetViewNormalFaces
        
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(0)
        IsSelected = SelectEntity(swFace, False, swView)
        Exit Sub
        
    End If
    
    If IsSelected Then
    
        Dim swSelectMgr As SldWorks.SelectionMgr
        Set swSelectMgr = swDrawing.SelectionManager
        
        Dim swComp As SldWorks.DrawingComponent
        Set swComp = swSelectMgr.GetSelectedObjectsComponent4(2, -1)
        
        If Not (Right(swComp.Name, Len(swComp.Name) - InStrRev(swComp.Name, "/")) = _
            Right(oComp.GetComponent.Name2, Len(oComp.GetComponent.Name2) - InStrRev(oComp.GetComponent.Name2, "/"))) Then
            
            Call SelectComponent(swDrawing, oComp, (oComp.xMax + oComp.xMin) / 2, YPos, Count + 1, IsSelected, swView)
            
        End If
        
    Else
    
        Call SelectComponent(swDrawing, oComp, (oComp.xMax + oComp.xMin) / 2, YPos, Count + 1, IsSelected, swView)
        
    End If
    
    
End Sub

Function GetViewName(wallName As String)

    Select Case wallName
        
        Case "Wall-A"
            
            GetViewName = "*Back"
        
        Case "Wall-B"
            
            GetViewName = "*Right"
        
        Case "Wall-C"
        
            GetViewName = "*Front"
        
        Case "Wall-D"
            
            GetViewName = "*Left"
            
        Case "Ceiling"
            
            GetViewName = "*Top"
            
        Case Else
            
            ViewNameForm.Show
            GetViewName = ViewNameForm.ViewNameBox.Value
            Unload ViewNameForm
    
    End Select
    
End Function

Function GetViewVector(viewName As String) As Double()

    Dim vViewRotation As Variant
    vViewRotation = swTopLevelModel.Extension.GetNamedViewRotation(viewName)
    
    Dim swMathVector As SldWorks.MathVector
    Set swMathVector = swMathUtility.CreateVector(zDirectionVector)
    
    Dim vTransformData(15) As Double
    vTransformData(0) = vViewRotation(0)
    vTransformData(1) = vViewRotation(1)
    vTransformData(2) = vViewRotation(2)
    vTransformData(3) = vViewRotation(3)
    vTransformData(4) = vViewRotation(4)
    vTransformData(5) = vViewRotation(5)
    vTransformData(6) = vViewRotation(6)
    vTransformData(7) = vViewRotation(7)
    vTransformData(8) = vViewRotation(8)
    'vTransformData(15) = 1
    
    Dim swMathTransform As SldWorks.MathTransform
    Set swMathTransform = swMathUtility.CreateTransform(vTransformData)
    
    Set swMathVector = swMathVector.MultiplyTransform(swMathTransform.Inverse)
    

    Select Case viewName
        
        Case "*Back"
            
            GetViewVector = GetOppositeVector(zDirectionVector)
        
        Case "*Right"

            GetViewVector = xDirectionVector
        
        Case "*Front"
        
            GetViewVector = zDirectionVector
        
        Case "*Left"
            
            GetViewVector = GetOppositeVector(xDirectionVector)
            
        Case "*Top"
            
            GetViewVector = yDirectionVector
            
        Case "*Bottom"
            
            GetViewVector = GetOppositeVector(yDirectionVector)
            
        Case Else
            
            

           GetViewVector = GetOppositeVector(swMathVector.ArrayData)
           
           
    
    End Select
       
End Function

Function ScaleAndInsertBottomView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            ViewWidth As Double, ViewHeight As Double) As SldWorks.View
            

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
    
    Dim IsViewSelected As Boolean
    
    Dim swDrawingModel As SldWorks.ModelDoc2
    Set swDrawingModel = swDrawing
    
    IsViewSelected = swDrawingModel.Extension.SelectByID2(swView.Name, "DRAWINGVIEW", 0, 0, 0, False, 0, Nothing, 0)
    Set ScaleAndInsertBottomView = swDrawing.CreateUnfoldedViewAt3(0.21593179, 0.08, 0, False)

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

Function GetComponentsSortedWithYPosition(swView As SldWorks.View, swDrawing As SldWorks.ModelDoc2, _
            swViewNormalVector As SldWorks.MathVector, ByRef ViewWidth As Double, ByRef ViewHeight As Double, _
                ByRef MaxHeightComp As IComp, ByRef IsZChannelExists As Boolean, ByRef zChannelList As IArrListObject, _
                    ByRef cChannelList As IArrListObject, ByRef lAngleList As IArrListObject) As IArrListObject
    
    swDrawing.ActivateView swView.Name

    Dim vComps As Variant
    vComps = swView.GetVisibleComponents()
    
    IsZChannelExists = False
    
    Dim swTopLevelAssy As SldWorks.AssemblyDoc
    Set swTopLevelAssy = swTopLevelModel

    Dim CompList As IArrListObject
    Set CompList = New IArrListObject

    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)
        
        Dim swCompFromRoot As SldWorks.Component2
        Set swCompFromRoot = GetComponentFromRoot(swComp.Name2, swComp, swTopLevelAssy)
            
        If swCompFromRoot.GetSuppression = swComponentSuppressionState_e.swComponentLightweight Then
            
            Dim bRet As Integer
            bRet = swCompFromRoot.SetSuppression2(swComponentSuppressionState_e.swComponentResolved)
            
        End If

        Dim swCompModel As SldWorks.ModelDoc2
        Set swCompModel = swCompFromRoot.GetModelDoc2

        If Not swCompModel Is Nothing Then
            
            Dim swCompProp As SldWorks.CustomPropertyManager
            Set swCompProp = swCompModel.Extension.CustomPropertyManager("")
            
            Dim Profile As String
            Dim ResolvedVal As String
            Dim wasResolved As Boolean
            swCompProp.Get5 "Profile", False, Profile, ResolvedVal, wasResolved
            
            If InStr(Profile, "EXT-") > 0 Then
            
                CompList.AddtoList GetComponentWithPosition(swCompFromRoot, swView, swDrawing, swViewNormalVector)
            
            ElseIf InStr(Profile, "Z-CHANNEL") > 0 Then
            
                zChannelList.AddtoList swCompFromRoot
                IsZChannelExists = True
                
            ElseIf InStr(Profile, "C-CHANNEL") > 0 Then
                'Debug.Print swCompFromRoot.Name2
                'Debug.Print swComp.Name2
                cChannelList.AddtoList swCompFromRoot
                
            ElseIf InStr(Profile, "L-ANGLE") > 0 Then
            
                lAngleList.AddtoList swCompFromRoot
                
            End If
        
        End If
        
    Next i
    
    CompList.SortItems "yMin", False
    
    Dim MinHeight As Double
    MinHeight = CompList.Items(LBound(CompList.Items)).yMin
    
    CompList.SortItems "xMin", False
    ViewWidth = CompList.Items(UBound(CompList.Items)).xMax - CompList.Items(LBound(CompList.Items)).xMin

    CompList.SortItems "yMax"
    Set MaxHeightComp = CompList.Items(LBound(CompList.Items))
    ViewHeight = MaxHeightComp.yMax - MinHeight
    
    Set GetComponentsSortedWithYPosition = CompList

End Function

Function GetComponentFromRoot(AssyName As String, swComp As SldWorks.Component2, swTopLevelAssy As SldWorks.AssemblyDoc) As SldWorks.Component2

        Dim compName As String
        compName = Right(AssyName, Len(AssyName) - InStrRev(AssyName, "/"))

        Dim TempComp As SldWorks.Component2
        Set TempComp = swTopLevelAssy.GetComponentByName(compName)

        If Not InStr(AssyName, TempComp.Name2) > 0 Then

            AssyName = Replace(AssyName, "/" & compName, "")

            Dim swAssy As SldWorks.Component2
            Set swAssy = GetComponentFromRoot(AssyName, swComp, swTopLevelAssy)

            Set GetComponentFromRoot = GetThisChildrenOfAssy(swAssy, compName)
            
        Else
        
            Set GetComponentFromRoot = TempComp

            
        End If
        
       
        
End Function

Function GetThisChildrenOfAssy(swAssy As SldWorks.Component2, compName As String) As SldWorks.Component2

    Dim vChild As Variant
    vChild = swAssy.GetChildren
    
    Dim i As Integer
    For i = LBound(vChild) To UBound(vChild)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vChild(i)
        
        Debug.Print swComp.Name2
        
        If InStr(swComp.Name2, compName) > 0 Then
        
            Set GetThisChildrenOfAssy = swComp
            Exit For
        
        End If
    
    Next i

End Function

Function GetComponentWithPosition(swComp As SldWorks.Component2, swView As SldWorks.View, _
        swDrawing As SldWorks.ModelDoc2, _
        swViewNormalVector As SldWorks.MathVector) As IComp

    Dim vFaces As Variant
    vFaces = GetComponentFaces(swComp) 'swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Face)
    
     'vEnts(0).GetBody

    If Not IsEmpty(vFaces) Then
    
        'Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))

        Dim vNormalFaces As Variant
        vNormalFaces = GetNormalFaces(vFaces, swComp.Transform2, swViewNormalVector)
        
        Dim MinPoint As Variant
        Dim MaxPoint As Variant
        Dim vBodyMinPoint(2) As Double
        Dim vBodyMaxPoint(2) As Double
        Call GetMinMaxBodyPointsInSheetSpace(swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, swView)
            
        Dim oComp As IComp
        Set oComp = New IComp
        oComp.Initialize swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, vNormalFaces
        
    End If
    
    'Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))
    
    Set GetComponentWithPosition = oComp

End Function

Private Sub GetMinMaxBodyPointsInSheetSpace(swComp As SldWorks.Component2, _
        ByRef MinPoint As Variant, ByRef MaxPoint As Variant, ByRef vBodyMinPoint() As Double, _
            ByRef vBodyMaxPoint() As Double, swView As SldWorks.View, Optional IsCorZ As Boolean = False)
            
'    Debug.Print swComp.GetModelDoc2.ConfigurationManager.ActiveConfiguration.Name
'    Debug.Print swComp.ReferencedConfiguration
'
'    If Not swComp.GetModelDoc2.ConfigurationManager.ActiveConfiguration.Name = swComp.ReferencedConfiguration Then
'
'        Debug.Print "no"
'
'    End If
'

    Dim vBodies As Variant

    If IsCorZ Then

        Dim vBodyInfo As Variant
        vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodyInfo)

    Else

        vBodies = swComp.GetModelDoc2.GetBodies(swSolidBody)
        
    End If
    
    Dim swBody As SldWorks.Body2
    Set swBody = vBodies(0)

    Dim vBodyBounds As Variant
    vBodyBounds = swBody.GetBodyBox

    vBodyMinPoint(0) = vBodyBounds(0)
    vBodyMinPoint(1) = vBodyBounds(1)
    vBodyMinPoint(2) = vBodyBounds(2)
            
    vBodyMaxPoint(0) = vBodyBounds(3)
    vBodyMaxPoint(1) = vBodyBounds(4)
    vBodyMaxPoint(2) = vBodyBounds(5)
    
    'Debug.Print swComp.Name2
            
    MinPoint = GetComponentPointInSheetSpace(swComp, vBodyMinPoint, swView)
    MaxPoint = GetComponentPointInSheetSpace(swComp, vBodyMaxPoint, swView)
    
End Sub

Function GetComponentFaces(swComp As SldWorks.Component2)
    
    Dim TempFaces As Variant
    
    Dim vBodies As Variant
    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, swBodyInfo_e.swNormalBody_e)
    
    Dim i As Integer
    Dim j As Integer
    For i = LBound(vBodies) To UBound(vBodies)
    
        Dim swBody As SldWorks.Body2
        Set swBody = vBodies(i)
        
        Dim vFaces As Variant
        vFaces = swBody.GetFaces
        
        If i = 0 Then
            
            TempFaces = vFaces
            
        Else
            
            TempFaces = CombineArr(TempFaces, vFaces)
            
        End If
    
    Next i
    
    GetComponentFaces = TempFaces

End Function

Function GetComponentEdges(swComp As SldWorks.Component2)
    
    Dim TempEdges As Variant
    
    Dim vBodies As Variant
    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, swBodyInfo_e.swNormalBody_e)
    
    Dim i As Integer
    Dim j As Integer
    For i = LBound(vBodies) To UBound(vBodies)
    
        Dim swBody As SldWorks.Body2
        Set swBody = vBodies(i)
        
        Dim vEdges As Variant
        vEdges = swBody.GetEdges
        
        If i = 0 Then
            
            TempEdges = vEdges
            
        Else
            
            TempEdges = CombineArr(TempEdges, vEdges)
            
        End If
    
    Next i
    
    GetComponentEdges = TempEdges

End Function

Function CombineArr(ByVal MainArr As Variant, ArrToAdd As Variant)

    Dim i As Integer
    For i = LBound(ArrToAdd) To UBound(ArrToAdd)
    
        ReDim Preserve MainArr(UBound(MainArr) + 1)
        Set MainArr(UBound(MainArr)) = ArrToAdd(i)
        
    Next i
    
    CombineArr = MainArr
    
End Function

Function GetNormalFaces(vFaces As Variant, CompTransform As IMathTransform, _
    swViewNormalVector As SldWorks.MathVector) As Variant
    
    Dim FaceCount As Integer
    FaceCount = 0
    
    Dim NormalFaces() As SldWorks.Face2

    Dim i As Integer
    For i = LBound(vFaces) To UBound(vFaces)

        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)

        Dim swSurface As SldWorks.Surface
        Set swSurface = swFace.GetSurface
        
        Set swViewNormalVector = swViewNormalVector.Normalise
        
        Dim swFaceNormalVector As SldWorks.MathVector
        Set swFaceNormalVector = swMathUtility.CreateVector(swFace.Normal)
        
        Set swFaceNormalVector = swFaceNormalVector.MultiplyTransform(CompTransform)
        Set swFaceNormalVector = swFaceNormalVector.Normalise
        
        Dim Angle As Double
        Dim DotProduct As Double
        DotProduct = swFaceNormalVector.Dot(swViewNormalVector)
        
        If DotProduct >= 1 Then
        
            Angle = Arccos(Int(DotProduct)) * 180# / 3.14159265359
            
        ElseIf DotProduct <= -1 Then
        
            Angle = Arccos(Int(DotProduct) + 1) * 180# / 3.14159265359
        
        Else
        
            Angle = Arccos(DotProduct) * 180# / 3.14159265359
            
        End If
 
        If swSurface.IsPlane And Angle <= 0.01 Then
            
            Dim swEnt As SldWorks.Entity
            Set swEnt = swFace
            Set swEnt = swEnt.GetSafeEntity
            
            ReDim Preserve NormalFaces(FaceCount)
            Set NormalFaces(FaceCount) = swEnt
            FaceCount = FaceCount + 1
            
        End If

    Next i
    
    GetNormalFaces = NormalFaces

End Function

Private Function GetComponentsSortedWithXPosition(vComps As Variant, ByRef FlatCompList As Variant, _
        swView As SldWorks.View, ByRef MaxCompHeight As Double) As Variant

    Dim CompWithPosDict As Scripting.Dictionary
    Set CompWithPosDict = New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim oComp As IComp
        Set oComp = vComps(i)

        Dim MinPoint As Variant
        MinPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
        Dim MaxPoint As Variant
        MaxPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
        
        oComp.UpdateSheetMaxMinDimensions swView, MinPoint, MaxPoint
        
        If i = LBound(vComps) Then
            
            MaxCompHeight = oComp.yMax
        
        End If

        Dim keyPosVal As Double
        keyPosVal = Round(oComp.xMin, 4)
        
        If CompWithPosDict.Exists(keyPosVal) Then
            
            Dim ExistingArr As Variant
            ExistingArr = CompWithPosDict(keyPosVal)
            
            ReDim Preserve ExistingArr(UBound(ExistingArr) + 1)
            Set ExistingArr(UBound(ExistingArr)) = oComp
            
            CompWithPosDict(keyPosVal) = ExistingArr
            
        Else
            
            Dim TempArr(0) As IComp
            Set TempArr(0) = oComp
            
            CompWithPosDict.Add keyPosVal, TempArr
            
        End If

    Next i

    GetComponentsSortedWithXPosition = GetDetailedCompList(CompWithPosDict, FlatCompList)
    
End Function

Private Function GetDetailedCompList(CompWithPosDict As Scripting.Dictionary, ByRef FlatCompList) As Variant

    Dim keysArrList As IArrList
    Set keysArrList = New IArrList
    
    keysArrList.AddItems = CompWithPosDict.Keys
    keysArrList.SortItems False
    
    Dim vItemsPos As Variant
    vItemsPos = keysArrList.Items
    
    Dim TempArr() As Variant
    
    Dim i As Integer
    For i = LBound(vItemsPos) To UBound(vItemsPos)
    
        If Not i = LBound(vItemsPos) Then
        
            If Abs(vItemsPos(i - 1) - vItemsPos(i)) <= 0.001 Then
                
                Dim samePosTempArr As Variant
                samePosTempArr = TempArr(UBound(TempArr))
                
                samePosTempArr = CombineArr(samePosTempArr, CompWithPosDict(vItemsPos(i)))
                
                TempArr(UBound(TempArr)) = samePosTempArr

            Else
            
                ReDim Preserve TempArr(UBound(TempArr) + 1)
                TempArr(UBound(TempArr)) = CompWithPosDict(vItemsPos(i))
            
            End If
            
        Else
        
            ReDim Preserve TempArr(i)
            TempArr(i) = CompWithPosDict(vItemsPos(i))
            
            
        End If
        
    Next i
    
    For i = LBound(TempArr) To UBound(TempArr)
    
          If i = 0 Then
                
            FlatCompList = TempArr(i)
                
        Else
            
            FlatCompList = CombineArr(FlatCompList, TempArr(i))
            
        End If
        
    Next i
    
    GetDetailedCompList = TempArr
    
End Function

Private Function GetConsolidatedList(vCompsOfComps As Variant, ByRef DoorOrHVACList As IArrListObject) As Variant

    Dim vConsolidatedLists As Variant
    Dim List As IConsolidatedList

    Dim IsInit As Boolean
    IsInit = True

    Dim DoororHVACSubAssy As IDoorOrHVACAssy
    
    Dim IsHVACStarted As Boolean
    IsHVACStarted = False
    
    Dim IsDoorStarted As Boolean
    IsDoorStarted = False
    
    Dim StartIndex As Integer
    Dim EndIndex As Integer
    
    Dim LastComp As IComp
        
    Dim i As Integer
    
    For i = LBound(vCompsOfComps) To UBound(vCompsOfComps)
        
        Dim vComps As Variant
        vComps = vCompsOfComps(i)
        
        If UBound(vComps) = 0 Then
        
            If IsHVACStarted Then
            
                IsHVACStarted = False
                Set DoororHVACSubAssy.EndComp = vComps(0)
                DoororHVACSubAssy.IsDoor = False
                EndIndex = i - 1
                DoorOrHVACList.AddtoList DoororHVACSubAssy
                Call UpdatedConsolidatedList(vConsolidatedLists, IsInit, StartIndex, EndIndex, vCompsOfComps)
                
            End If
        
            Dim oComp As IComp
            Set oComp = vComps(0)
            
            If IsInit Then
                
                Set List = New IConsolidatedList
                Set List.Comp = oComp
    
                ReDim vConsolidatedLists(0)
                Set vConsolidatedLists(0) = List
                
                IsInit = False
                
            Else
                
                Dim LastConsolidatedItem As IConsolidatedList
                Set LastConsolidatedItem = vConsolidatedLists(UBound(vConsolidatedLists))

                If LastConsolidatedItem.Comp.GetPathName = oComp.GetPathName Then

                    LastConsolidatedItem.IncQty

                Else

                    Set List = New IConsolidatedList
                    Set List.Comp = oComp
    
                    ReDim Preserve vConsolidatedLists(UBound(vConsolidatedLists) + 1)
                    Set vConsolidatedLists(UBound(vConsolidatedLists)) = List
    
                End If
                
            End If
            
            If Not LastComp Is Nothing Then
            
                If Not Abs(LastComp.yMin - oComp.yMin) <= 0.00001 Then

                    If IsDoorStarted Then
                
                        IsDoorStarted = False
                        Set DoororHVACSubAssy.EndComp = oComp
                        DoororHVACSubAssy.DoororHVACWidth = oComp.xMin - DoororHVACSubAssy.StartComp.xMax
                        DoororHVACSubAssy.DoororHVACLength = Abs(LastComp.yMin - oComp.yMin)
                        
                        DoororHVACSubAssy.IsDoor = True
                        
                        DoorOrHVACList.AddtoList DoororHVACSubAssy
                        
                    Else
                    
                        IsDoorStarted = True
                        Set DoororHVACSubAssy = New IDoorOrHVACAssy
                        Set DoororHVACSubAssy.StartComp = LastComp

                    End If
                    
                End If
                
            End If
        
            Set LastComp = vComps(0)

        Else
            
            If False = IsHVACStarted Then
                
                IsHVACStarted = True
                StartIndex = i
                Set DoororHVACSubAssy = New IDoorOrHVACAssy
                Set DoororHVACSubAssy.StartComp = vCompsOfComps(i - 1)(0)
                
            End If

        End If

    Next i

    GetConsolidatedList = vConsolidatedLists

End Function

Private Sub UpdatedConsolidatedList(ByRef vConsolidatedLists As Variant, IsInit As Boolean, StartIndex As Integer, _
                EndIndex As Integer, vCompsOfComps As Variant)
                
    Dim i As Integer
    
    Dim TempDictComp As Scripting.Dictionary
    Set TempDictComp = New Scripting.Dictionary
    
    Dim TempDictQty As Scripting.Dictionary
    Set TempDictQty = New Scripting.Dictionary
    
    For i = StartIndex To EndIndex
    
        Dim vComps As Variant
        vComps = vCompsOfComps(i)
        
        Dim j As Integer
        For j = LBound(vComps) To UBound(vComps)
        
            Dim oComp As IComp
            Set oComp = vComps(j)
            
            If Not (j = LBound(vComps)) Then
                
                oComp.IsTop = False
            
            End If
            
            If j = UBound(vComps) Then
                
                oComp.IsBottom = True
                
            End If
            
            Dim PathVal As String
            PathVal = oComp.GetComponent.GetPathName
            
            If TempDictComp.Exists(PathVal) Then
            
                TempDictQty.Item(PathVal) = TempDictQty.Item(PathVal) + 1
                
            Else
            
                TempDictComp.Add PathVal, oComp
                TempDictQty.Add PathVal, 1
                
            End If
        
        Next j
        
    Next i
    
    Dim DictKeys As Variant
    DictKeys = TempDictComp.Keys
    
    Dim k As Integer
    For k = LBound(DictKeys) To UBound(DictKeys)
    
        Dim List As IConsolidatedList
        Set List = New IConsolidatedList
        Set List.Comp = TempDictComp.Item(DictKeys(k))
        List.Qty = TempDictQty.Item(DictKeys(k))
        
        If IsInit Then

            ReDim vConsolidatedLists(0)
            Set vConsolidatedLists(0) = List
                
            IsInit = False
            
        Else
        
            ReDim Preserve vConsolidatedLists(UBound(vConsolidatedLists) + 1)
            Set vConsolidatedLists(UBound(vConsolidatedLists)) = List
        
        End If

    Next k

End Sub




