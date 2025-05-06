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
Const SheetBorderTop As Double = 0.27030866

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

    SelectEntity = swView.SelectEntity(swEnt, Append)
    
End Function

Private Sub CreateButton_Click()

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

    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\ASSEMBLY DRAWING.drwdot", 0, 0, 0)

    Set swSketchMgr = swDrawing.SketchManager

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)

    Dim swFrontView As SldWorks.View
    Set swFrontView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), viewName, 0.21593179, 0.15772741, 0)
    
    Dim IsViewSelected As Boolean
    IsViewSelected = swDrawing.Extension.SelectByID2(swFrontView.Name, "DRAWINGVIEW", 0, 0, 0, False, 0, Nothing, 0)
    
    Dim swBottomView As SldWorks.View
    Set swBottomView = swDrawing.CreateUnfoldedViewAt3(0.21593179, 0.065, 0, False)

    Dim CompList As IArrListObject
    Set CompList = GetComponentsSortedWithXPosition(swFrontView, swDrawing, "EXT-")
    
    Dim InternalCompList As IArrListObject
    Set InternalCompList = GetComponentsSortedWithXPosition(swFrontView, swDrawing, "INT-")
    
    Dim ViewWidth As Double
    Dim ViewHeight As Double
    Dim MaxHeightComp As IComp
                

    If IsEmpty(CompList.Items) Then
    
        Set CompList = GetComponentsSortedWithXPosition(swBottomView, swDrawing, "EXT-")
        Call UpdateMaxMinPoints(CompList.Items, swFrontView)
        Call GetComponentBoundsInView(CompList, ViewWidth, ViewHeight, MaxHeightComp)
         
    Else
    
        Call GetComponentBoundsInView(CompList, ViewWidth, ViewHeight, MaxHeightComp)

    End If
                

    Dim IsMultipleAssembly As Boolean
    IsMultipleAssembly = CheckForMultipleAssembly(ViewWidth / swFrontView.ScaleDecimal, ViewHeight / swFrontView.ScaleDecimal)
    
    Call ActivateDrawingDocument(swTopLevelModel)
    
    Dim subAssyEndComponents As Variant
    If IsMultipleAssembly Then

        SubAssyForm.Show vbModeless

        IsSubAssyFormClicked = False
        Do While (IsSubAssyFormClicked = False)

            DoEvents

        Loop

        subAssyEndComponents = GetSelectedComponents

    End If
    
    Dim InsulationComponents As Variant
    InsulationForm.Show vbModeless
    IsInsulationFormClicked = False
    Do While IsInsulationFormClicked = False
        DoEvents
    Loop
    InsulationComponents = GetSelectedComponents
    
    Call ActivateDrawingDocument(swDrawing)
    

    Call ScaleView(swDrawing, swFrontView, ViewWidth, ViewHeight)
    
    Call UpdateMaxMinPoints(CompList.Items, swFrontView)
    Call UpdateMaxMinPoints(InternalCompList.Items, swFrontView)
    
    Call UpdateBottomViewPosition(InternalCompList.Items, swDrawing, swBottomView)
    
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
    
    Dim vConsolidatedList As Variant
    Dim DoorList As IArrListObject
    vConsolidatedList = GetConsolidatedList(InternalCompList, DoorList)
    
    Call AddCallouts(vConsolidatedList, swDrawing, swFrontView, MaxHeightComp.yMax)

    Dim swLeftEdge As SldWorks.Edge
    Dim swRightEdge As SldWorks.Edge
    
    Dim swBottomEdge As SldWorks.Edge
    Set swBottomEdge = AddDimensionInFrontView(swFrontView, CompList.Items, MaxHeightComp, swDrawing, swLeftEdge, swRightEdge)
        
    Dim FlatCompDict As Scripting.Dictionary
    Dim CompNoDict As New Scripting.Dictionary
    Set FlatCompDict = GetCompDictionary(CompList.Items, CompNoDict)

    Dim lTabList As IArrListObject
    Set lTabList = GetLTabList(InternalCompList, swFrontView)
    
    Dim TabAssyList As IArrListObject
    Set TabAssyList = New IArrListObject
    
    If Not IsEmpty(lTabList.Items) Then
    
        Dim xTabDict As Scripting.Dictionary
        Set xTabDict = GetConsolidatedTabListBasedOnXPos(lTabList)

        Set TabAssyList = GetTabAssList(xTabDict)
        
    End If
    
    Dim subAssylist As IArrListObject
    Set subAssylist = New IArrListObject
    
    If Not IsEmpty(subAssyEndComponents) Then
    
        Dim vSubAssyComponentsIdx As Variant
        vSubAssyComponentsIdx = GetSubAssyComponentsIndexSorted(subAssyEndComponents, CompNoDict)
    
        Set subAssylist = AddSplitLines(vSubAssyComponentsIdx, swDrawing, swFrontView, FlatCompDict, CompNoDict, True, swLeftEdge, swRightEdge, False)
        Call AddSplitLines(vSubAssyComponentsIdx, swDrawing, swBottomView, FlatCompDict, CompNoDict, False, swLeftEdge, swRightEdge)
        
        Call CheckForLTabOrDoorAssy(subAssylist, TabAssyList)
        Call CheckForLTabOrDoorAssy(subAssylist, DoorList)

    End If
    
    Dim oSubAssy As ISubAssy
    Set oSubAssy = New ISubAssy
    
    Set oSubAssy.StartComp = FlatCompDict.Items(0)
    Set oSubAssy.EndComp = FlatCompDict.Items(UBound(FlatCompDict.Items))
    Set oSubAssy.StartEdge = swLeftEdge
    Set oSubAssy.EndEdge = swRightEdge
    oSubAssy.StartIdx = 0
    oSubAssy.EndIdx = UBound(FlatCompDict.Items)
    oSubAssy.AddEntireList TabAssyList
    oSubAssy.AddEntireList DoorList

    subAssylist.AddtoList oSubAssy

    Dim swLeftSketch As SldWorks.SketchSegment
    Dim swRightSketch As SldWorks.SketchSegment

    Dim MaxClearance As Double
    
    Call SketchLineForNonCornerPanels(swFrontView, wallName, swDrawing, oSubAssy, swBottomEdge, MaxClearance, swLeftSketch, swRightSketch)
    Call AddDimensionsForLTabOrDoorInEachSubAssy(subAssylist, swDrawing, swFrontView, MaxClearance, swLeftSketch)
    Call AddDimensionFromCornerSketches(swDrawing, swFrontView, swLeftSketch, swRightSketch, oSubAssy, MaxClearance)
    
    
    Call AddVerticalDimensionForLTab(swDrawing, swFrontView, swBottomEdge, TabAssyList)
    Call AddVerticalDimensionForDoor(swDrawing, swFrontView, swBottomEdge, DoorList, oSubAssy)
    Call AddCrossMarkForDoor(swDrawing, swFrontView, swBottomEdge, DoorList)
    Call CleanUpActivateAndAddViewLabel(swDrawing, swFrontView, wallName, oSubAssy.StartComp.yMin - MaxClearance - 0.005, (oSubAssy.StartComp.xMin + oSubAssy.EndComp.xMax) / 2)
    
     Call AddCastingSketchAndNote(oSubAssy.EndComp, swBottomView, swSketchMgr, swDrawing)
     Call InsulationHatchesAndCallouts(InsulationComponents, swDrawing, swBottomView)
     
     Call UpdateFrontViewPosition(InternalCompList.Items, swDrawing, swFrontView)


    'Dim NoteCount As Integer
    'Call AddStructuralNotes(swDrawing, swSheet, wallName)

    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True

    Unload Me
    
End Sub

Sub InsulationHatchesAndCallouts(vInsulationComps, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    If Not IsEmpty(vInsulationComps) Then
    
        Dim i As Integer
        Dim Clearance As Double
        Clearance = 0.005
        For i = LBound(vInsulationComps) To UBound(vInsulationComps)
            
            Dim swInsulationComp As SldWorks.Component2
            Set swInsulationComp = vInsulationComps(i)
            
            Clearance = Clearance + 0.005
            
            Call AddInsulationMaterialNote(swInsulationComp, swView, swDrawing, Clearance)
            Call AddInsulationHatches(swInsulationComp, swView, swDrawing)
            
            
        Next i
    
    End If

End Sub
Sub AddCrossMarkForDoor(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
                swBottomEdge As SldWorks.Edge, DoorList As IArrListObject)

    Dim vDoorItems As Variant
    vDoorItems = DoorList.Items
    
    If Not IsEmpty(vDoorItems) Then
    
        swDrawing.ActivateSheet swDrawing.GetCurrentSheet.GetName
        swDrawing.ActivateView swView.Name
        
        swView.FocusLocked = True
        
    
        Dim i As Integer
        For i = LBound(vDoorItems) To UBound(vDoorItems)
        
            Dim oDoorAssy As IDoorAssy
            Set oDoorAssy = vDoorItems(i)
            
            Dim DoorLeftEdge As SldWorks.Edge
            Set DoorLeftEdge = GetEdgeInView(oDoorAssy.StartComp, swView, False, True)
            
            Dim DoorRightEdge As SldWorks.Edge
            Set DoorRightEdge = GetEdgeInView(oDoorAssy.EndComp, swView, False, False)
            
            Dim DoorBottomEdge As SldWorks.Edge
            Set DoorBottomEdge = GetEdgeInView(oDoorAssy.StartComp, swView, True, False)

            Dim DoorTopEdge As SldWorks.Edge
            Set DoorTopEdge = GetEdgeInView(oDoorAssy.TopComp, swView, True, False)
            
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
            UpperLeftPoint(1) = oDoorAssy.TopComp.yMin
            UpperLeftPoint(2) = 0
            
            Dim vUpperLeftPoint As Variant
            vUpperLeftPoint = GetSheetPointInViewSpace(swView, UpperLeftPoint)

            Dim UpperRightPoint(2) As Double
            UpperRightPoint(0) = oDoorAssy.EndComp.xMin
            UpperRightPoint(1) = oDoorAssy.TopComp.yMin
            UpperRightPoint(2) = 0
            
            Dim vUpperRightPoint As Variant
            vUpperRightPoint = GetSheetPointInViewSpace(swView, UpperRightPoint)
            
            Dim swSketchManager As SldWorks.SketchManager
            Set swSketchManager = swDrawing.SketchManager
            
            Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerLeftPoint, vUpperRightPoint, DoorLeftEdge, DoorRightEdge, DoorBottomEdge, DoorTopEdge)
            Call CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing, swView, vLowerRightPoint, vUpperLeftPoint, DoorRightEdge, DoorLeftEdge, DoorBottomEdge, DoorTopEdge)

        Next i

    End If

    
End Sub
Sub CreateSketchSegmentAndAddRelation(swSketchManager, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, FirstPoint As Variant, SecondPoint As Variant, FirstPtVerticalEdge As SldWorks.Edge, _
                SecondPtVerticalEdge As SldWorks.Edge, FirstPtHorEdge As SldWorks.Edge, SecondPtHorEdge As SldWorks.Edge)
    
    
    
    Dim swSketchSegment As SketchSegment
    Set swSketchSegment = swSketchManager.CreateLine(FirstPoint(0), FirstPoint(1), FirstPoint(2), _
                        SecondPoint(0), SecondPoint(1), SecondPoint(2))
    swSketchSegment.ConstructionGeometry = True
    
    If Not swSketchSegment Is Nothing Then
        
        Dim swSketchLine As SldWorks.SketchLine
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

Sub AddManualParanthesis(swDisplayDim As SldWorks.DisplayDimension, Optional Qty As Integer = 1, Optional IsTyp As Boolean = False)

    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, "("
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ")"

    If Qty > 1 Then
    
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X ("
    
    End If
    
    If IsTyp Then
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ") TYP."
        
    End If

End Sub

Sub AddVerticalDimensionForDoor(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
                            swBottomEdge As SldWorks.Edge, DoorList As IArrListObject, oSubAssy As ISubAssy)
    
    swDrawing.ActivateView swView.Name
    If Not IsEmpty(DoorList.Items) Then
    
        Dim vDoorItems As Variant
        vDoorItems = DoorList.Items

        Dim ConsolidatedDoorDict As Scripting.Dictionary
        Set ConsolidatedDoorDict = New Scripting.Dictionary
        
        Dim ConsolidatedQtyDict As Scripting.Dictionary
        Set ConsolidatedQtyDict = New Scripting.Dictionary
        
        Dim i As Integer
        For i = LBound(vDoorItems) To UBound(vDoorItems)
            
            Dim oDoorAssy As IDoorAssy
            Set oDoorAssy = vDoorItems(i)
            
            Dim yDiff As Double
            yDiff = Round(Abs(oSubAssy.yMin - oDoorAssy.TopComp.yMin), 3)
            
            If ConsolidatedDoorDict.Exists(yDiff) Then
            
                ConsolidatedQtyDict.Item(yDiff) = ConsolidatedQtyDict.Item(yDiff) + 1
            
            Else
            
                Dim swDoorTopEdge As SldWorks.Edge
                Set swDoorTopEdge = GetEdgeInView(oDoorAssy.TopComp, swView, True, False)
                
                Dim swDisplayDim As SldWorks.DisplayDimension
                Set swDisplayDim = SelectAndAddDimension(swBottomEdge, swDoorTopEdge, swDrawing, _
                                oDoorAssy.EndComp.xMin + 0.01, oDoorAssy.TopComp.yMin - 0.01, swView, False, False, IsHorizontalDim:=False)
                
                ConsolidatedDoorDict.add yDiff, swDisplayDim
                ConsolidatedQtyDict.add yDiff, 1

            End If
        
        Next i
        
        Call AddQtyToDoorDimension(ConsolidatedDoorDict, ConsolidatedQtyDict)
        
    End If
     
End Sub

Sub AddQtyToDoorDimension(ConsolidatedDoorDict As Scripting.Dictionary, ConsolidatedQtyDict As Scripting.Dictionary)

    Dim vKeys As Variant
    vKeys = ConsolidatedDoorDict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
        
        Dim Qty As Integer
        Qty = ConsolidatedQtyDict.Item(vKeys(i))
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = ConsolidatedDoorDict.Item(vKeys(i))
        
        If Not swDisplayDim Is Nothing Then
        
            If Qty > 1 Then
    
                Call AddManualParanthesis(swDisplayDim, Qty)
            
            Else
            
                swDisplayDim.ShowParenthesis = True
                
            End If
            
        End If
    
    Next i
    
End Sub
            
Sub AddVerticalDimensionForLTab(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            swBottomEdge As SldWorks.Edge, TabAssyList As IArrListObject)
            
    'TabAssyList.SortItems "LowerLeftYPoint", False
    
    If Not IsEmpty(TabAssyList.Items) Then
    
        Dim vTabAssy As Variant
        vTabAssy = TabAssyList.Items
        
        Dim i As Integer
        For i = LBound(vTabAssy) To UBound(vTabAssy)
        
            Dim oTabAssy As ILTabAssy
            Set oTabAssy = vTabAssy(i)
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            
            If i = LBound(vTabAssy) Then
            
                Call AddLTabVerticalPosDimension(oTabAssy, swBottomEdge, swDrawing, swView)
                Call AddLTabVerticalDimension(oTabAssy, swDrawing, swView)
            

            Else
            
                Dim PrevTabAssy As ILTabAssy
                Set PrevTabAssy = vTabAssy(i - 1)

                If False = IsAnyPreviousTabAssySame(oTabAssy, i - 1, vTabAssy) Then

                        
                    If (Abs(PrevTabAssy.LowerLeftXPoint - oTabAssy.LowerLeftXPoint) <= 0.001 And _
                            Abs(PrevTabAssy.UpperRightXPoint - oTabAssy.UpperRightXPoint) <= 0.001) Then
                           
                        Set swDisplayDim = SelectAndAddDimension(PrevTabAssy.UpperRightLTab.EndHorizontalEdge, oTabAssy.LowerRightLTab.EndHorizontalEdge, swDrawing, _
                                oTabAssy.UpperRightXPoint + 0.01, oTabAssy.LowerLeftYPoint - 0.01, swView, False, IsHorizontalDim:=False)
                                
                        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "TYP."
                        
                    Else
                                        
                        Call AddLTabVerticalPosDimension(oTabAssy, swBottomEdge, swDrawing, swView)
  
                    End If
                    
                    Call AddLTabVerticalDimension(oTabAssy, swDrawing, swView, -1)
                        
                End If
                    

                
            End If

        Next i

    End If

End Sub

Function IsAnyPreviousTabAssySame(oTabAssyToCheck As ILTabAssy, Idx As Integer, vTabAssy As Variant) As Boolean
    
    IsAnyPreviousTabAssySame = False
    
    Dim i As Integer
    For i = LBound(vTabAssy) To Idx
        
        Dim oTabAssy As ILTabAssy
        Set oTabAssy = vTabAssy(i)
        
        If (Abs(oTabAssyToCheck.LowerLeftYPoint - oTabAssy.LowerLeftYPoint) <= 0.001 And _
                           Abs(oTabAssyToCheck.UpperRightYPoint - oTabAssy.UpperRightYPoint) <= 0.001) Then
                           
            IsAnyPreviousTabAssySame = True
            Exit For
            
        End If
        
    
    Next i
    
End Function

Sub AddLTabVerticalPosDimension(oTabAssy As ILTabAssy, swBottomEdge As SldWorks.Edge, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swDisplayDim As SldWorks.DisplayDimension
    Set swDisplayDim = SelectAndAddDimension(swBottomEdge, oTabAssy.LowerRightLTab.EndHorizontalEdge, swDrawing, _
                        oTabAssy.UpperRightXPoint + 0.01, oTabAssy.LowerLeftYPoint - 0.01, swView, False, IsHorizontalDim:=False)
                        
                        
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "TYP."
    
End Sub


Sub AddLTabVerticalDimension(oLTabAssy As ILTabAssy, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, Optional Reverse As Integer = 1)

    Dim swDisplayDim As SldWorks.DisplayDimension
                  
    Set swDisplayDim = SelectAndAddDimension(oLTabAssy.LowerRightLTab.EndHorizontalEdge, _
                            oLTabAssy.UpperRightLTab.EndHorizontalEdge, swDrawing, _
                        oLTabAssy.UpperRightXPoint + 0.01, oLTabAssy.UpperRightYPoint - (0.00625 * Reverse), swView, False, IsHorizontalDim:=False)
                        
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "TYP."
    
End Sub
Private Sub AddDimensionsForLTabOrDoorInEachSubAssy(subAssylist As IArrListObject, swDrawing As SldWorks.DrawingDoc, _
            swView As SldWorks.View, MaxClearance As Double, swLeftSketch As SldWorks.SketchSegment)

    Dim vSubAssy As Variant
    vSubAssy = subAssylist.Items

    MaxClearance = 0
    
    Dim i As Integer
    For i = LBound(vSubAssy) To UBound(vSubAssy)
    
        Dim oSubAssy As ISubAssy
        Set oSubAssy = vSubAssy(i)
        
        Dim Clearance As Double
        Clearance = 0
 
        
        If i < UBound(vSubAssy) Then

            Call AddDimensionsForLTabOrDoor(oSubAssy.GetLTabOrDoorList, oSubAssy, swDrawing, swView, Clearance)
            
            If i = 0 Then
            
                Call AddLTabRefDimensionFromSketchEnd(oSubAssy, swDrawing, swView, swLeftSketch)
                
            End If
            
        Else
        
            If UBound(vSubAssy) = 0 Then
            
               Call AddDimensionsForLTabOrDoor(oSubAssy.GetLTabOrDoorList, oSubAssy, swDrawing, swView, Clearance)
               Call AddOverallDimension(oSubAssy, swDrawing, swView, Clearance)
               
            Else
            
                Call AddOverallDimension(oSubAssy, swDrawing, swView, MaxClearance)
               
            End If
        

            
        End If

        
        If Clearance > MaxClearance Then
        
            MaxClearance = Clearance
            
        End If
    
    Next i

End Sub

Sub AddLTabRefDimensionFromSketchEnd(oSubAssy As ISubAssy, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, swLeftSketch As SldWorks.SketchSegment)


    Dim vLTabItems As Variant
    vLTabItems = oSubAssy.GetLTabList
    
    If Not IsEmpty(vLTabItems) Then
    
        Dim ClearanceUp As Double
        ClearanceUp = 0.005
        
        Dim i As Integer
        For i = LBound(vLTabItems) To UBound(vLTabItems)
        
            Dim oLTabAssy As ILTabAssy
            Set oLTabAssy = vLTabItems(i)
            
            If UBound(vLTabItems) = 0 Then
                
                Call CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy, oSubAssy, ClearanceUp, swDrawing, swView, swLeftSketch)
                
            Else
                
                If i < UBound(vLTabItems) Then
                
                    Dim NextLTabAssy As ILTabAssy
                    Set NextLTabAssy = vLTabItems(i + 1)
                    
                    If Not Abs(oLTabAssy.LowerLeftXPoint - NextLTabAssy.LowerLeftXPoint) <= 0.001 Then
                    
                        Call CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy, oSubAssy, ClearanceUp, swDrawing, swView, swLeftSketch)
                        
                    End If
                    
                Else
                
                    Call CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy, oSubAssy, ClearanceUp, swDrawing, swView, swLeftSketch)
                    
                End If
                
            End If

        Next i
        
    End If
    

End Sub

Sub CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy As ILTabAssy, oSubAssy As ISubAssy, ByRef ClearanceUp As Double, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, swLeftSketch As SldWorks.SketchSegment)
        
    
        If Not swLeftSketch Is Nothing Then
        
            swLeftSketch.Select4 False, Nothing
            Call SelectEntity(oLTabAssy.UpperLeftLTab.EndVerticalEdge, True, swView)
            ClearanceUp = ClearanceUp + 0.007
            
        End If

        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.StartComp.xMin + 0.01, oSubAssy.yMax + ClearanceUp, 0)
        
        If Not swDisplayDim Is Nothing Then

            swDisplayDim.CenterText = True
            Call AddManualParanthesis(swDisplayDim, IsTyp:=True)
            
        End If


End Sub


Private Sub AddDimensionsForLTabOrDoor(vDoorOrTabItems As Variant, oSubAssy As ISubAssy, swDrawing As SldWorks.DrawingDoc, _
            swView As SldWorks.View, ByRef Clearance As Double)

    Dim j As Integer
    
    If Not IsEmpty(vDoorOrTabItems) Then
    
        Dim swDisplayDim As SldWorks.DisplayDimension
        
        For j = LBound(vDoorOrTabItems) To UBound(vDoorOrTabItems)
                
            Dim oDoorOrTabAssy As ITabOrDoorAssy
            Set oDoorOrTabAssy = vDoorOrTabItems(j)
    
            If oDoorOrTabAssy.IsDoor Then
            
                Clearance = Clearance + 0.006
                
                Dim oDoor As IDoorAssy
                Set oDoor = oDoorOrTabAssy.Doorassy
            
                Dim oStartComp As IComp
                Set oStartComp = oDoor.StartComp
                
                Dim swDoorStartEdge As SldWorks.Edge
                Set swDoorStartEdge = GetEdgeInView(oStartComp, swView, False, True)
                
                
                Set swDisplayDim = SelectAndAddDimension(oSubAssy.StartEdge, swDoorStartEdge, swDrawing, _
                        oStartComp.xMin + 0.01, oStartComp.yMin - Clearance, swView, False, True)
            
                Dim oEndComp As IComp
                Set oEndComp = oDoor.EndComp
                
                Dim swDoorEndEdge As SldWorks.Edge
                Set swDoorEndEdge = GetEdgeInView(oEndComp, swView, False, False)
                
                Set swDisplayDim = SelectAndAddDimension(swDoorStartEdge, swDoorEndEdge, swDrawing, _
                        oEndComp.xMin - 0.01, oStartComp.yMin - Clearance, swView, False, True)
                        
            Else
            
                Dim oLTabAssy As ILTabAssy
                Set oLTabAssy = oDoorOrTabAssy.TabAssy
                
                If j = LBound(vDoorOrTabItems) Then
                
                    Clearance = Clearance + 0.006
                    Call AddLTabPosDimension(oLTabAssy, oSubAssy, Clearance, swDrawing, swView)
                    Call AddLTabLengthDimension(oLTabAssy, swDrawing, swView)
                    'Call CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy, oSubAssy, ClearanceUp, swDrawing, swView, swLeftSketch, SubAssyIdx)
     
                       
                Else
                
                    If vDoorOrTabItems(j - 1).IsDoor Then
                    
                        Clearance = Clearance + 0.006
                        Call AddLTabPosDimension(oLTabAssy, oSubAssy, Clearance, swDrawing, swView)
                        Call AddLTabLengthDimension(oLTabAssy, swDrawing, swView)
                        'Call CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy, oSubAssy, ClearanceUp, swDrawing, swView, swLeftSketch, SubAssyIdx)
                        
                    Else
                
                        Dim PrevTabAssy As ILTabAssy
                        Set PrevTabAssy = vDoorOrTabItems(j - 1).TabAssy
                        
                        If Not Abs(PrevTabAssy.LowerLeftXPoint - oLTabAssy.LowerLeftXPoint) <= 0.001 Then
                        
                            Clearance = Clearance + 0.006
                            Call AddLTabPosDimension(oLTabAssy, oSubAssy, Clearance, swDrawing, swView)
                            Call AddLTabLengthDimension(oLTabAssy, swDrawing, swView)
                            'Call CheckAndAddLTabDimensionFromSketchEnd(oLTabAssy, oSubAssy, ClearanceUp, swDrawing, swView, swLeftSketch, SubAssyIdx)
                            
                        Else
                        
                            If Not Abs(PrevTabAssy.UpperRightXPoint - oLTabAssy.UpperRightXPoint) <= 0.001 Then
                            
                                Call AddLTabLengthDimension(oLTabAssy, swDrawing, swView)
                            
                            End If
                                
                        End If
                        
                    End If
                
                End If
            
            End If
            
            
                
        Next j
            
    End If
    


End Sub


Sub AddLTabPosDimension(oLTabAssy As ILTabAssy, oSubAssy As ISubAssy, Clearance As Double, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swDisplayDim As SldWorks.DisplayDimension
    Set swDisplayDim = SelectAndAddDimension(oLTabAssy.LowerLeftLTab.EndVerticalEdge, _
                        oSubAssy.StartEdge, swDrawing, _
                        oLTabAssy.LowerLeftXPoint - 0.01, oSubAssy.StartComp.yMin - Clearance, swView, False)
                        
                        
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "TYP."
    
End Sub


Sub AddLTabLengthDimension(oLTabAssy As ILTabAssy, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swDisplayDim As SldWorks.DisplayDimension
                  
    Set swDisplayDim = SelectAndAddDimension(oLTabAssy.LowerLeftLTab.EndVerticalEdge, _
                            oLTabAssy.LowerRightLTab.EndVerticalEdge, swDrawing, _
                        oLTabAssy.LowerLeftXPoint + 0.01, oLTabAssy.LowerLeftYPoint - 0.005, swView, False)
                        
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "TYP."
    
End Sub


Sub CheckForLTabOrDoorAssy(subAssylist As IArrListObject, TaborDoorAssyList As IArrListObject)

    If Not IsEmpty(TaborDoorAssyList.Items) Then
    
        Dim vTaborDoors As Variant
        vTaborDoors = TaborDoorAssyList.Items
        
        Dim vSubAssy As Variant
        vSubAssy = subAssylist.Items
        
        Dim LastFoundIdx As Integer
        LastFoundIdx = 0
        
        Dim i As Integer
        For i = LBound(vTaborDoors) To UBound(vTaborDoors)
        
            Dim TabOrDoorObj As Object
            Set TabOrDoorObj = vTaborDoors(i)

            Dim j As Integer
            For j = LastFoundIdx To UBound(vSubAssy)
            
                Dim oSubAssy As ISubAssy
                Set oSubAssy = vSubAssy(j)
                
                If TabOrDoorObj.LowerLeftXPoint > oSubAssy.StartComp.xMin And _
                         TabOrDoorObj.LowerLeftXPoint < oSubAssy.EndComp.xMax Then
                       
                    oSubAssy.AddToLTaborDoorList TabOrDoorObj
                    LastFoundIdx = j
                    Exit For
                    
                End If
            
            Next j
            
        Next i
    
    End If

End Sub


Function GetConsolidatedTabListBasedOnXPos(lTabList As IArrListObject) As Scripting.Dictionary
    
    Set GetConsolidatedTabListBasedOnXPos = New Scripting.Dictionary
    
    Dim vTabItems As Variant
    vTabItems = lTabList.Items
    
    Dim i As Integer
    For i = LBound(vTabItems) To UBound(vTabItems)
    
        Dim oLTab As ILTabs
        Set oLTab = vTabItems(i)
    
        If GetConsolidatedTabListBasedOnXPos.Exists(oLTab.xPoint) Then
            
            GetConsolidatedTabListBasedOnXPos.Item(oLTab.xPoint).AddtoList oLTab
        Else
            
            Dim TempLTabList As IArrListObject
            Set TempLTabList = New IArrListObject
            
            TempLTabList.AddtoList oLTab
            
            If GetConsolidatedTabListBasedOnXPos.Count = 0 Then
            
                GetConsolidatedTabListBasedOnXPos.add oLTab.xPoint, TempLTabList
                
            Else
                
                If Abs(GetConsolidatedTabListBasedOnXPos.Keys(UBound(GetConsolidatedTabListBasedOnXPos.Keys)) - oLTab.xPoint) <= 0.001 Then
                
                     GetConsolidatedTabListBasedOnXPos.Item(GetConsolidatedTabListBasedOnXPos.Keys(UBound(GetConsolidatedTabListBasedOnXPos.Keys))).AddtoList oLTab
                
                Else
                
                    GetConsolidatedTabListBasedOnXPos.add oLTab.xPoint, TempLTabList
                
                End If
                
            End If
            
        End If

    Next i
    
    'Call SortArrListInEachDictionary(GetConsolidatedTabListBasedOnXPos)
    
End Function

Function GetTabAssList(Dict As Scripting.Dictionary) As IArrListObject

    Set GetTabAssList = New IArrListObject

    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(i))
        
        ArrList.SortItems "yPoint", False
        
        
        Dim vLTabs As Variant
        vLTabs = ArrList.Items
        
        If (UBound(vLTabs) + 1) Mod 2 = 0 Then
        
            Dim j As Integer
            For j = LBound(vLTabs) To UBound(vLTabs) Step 2
                
                Dim oLTab As ILTabs
                Set oLTab = vLTabs(j)
                
                Dim oLTabNext As ILTabs
                Set oLTabNext = vLTabs(j + 1)
                
                If oLTab.IsLeft And oLTabNext.IsLeft Then
                
                    If oLTab.IsBottom And Not (oLTabNext.IsBottom) Then
                    
                        Dim oLTabAssy As ILTabAssy
                        Set oLTabAssy = New ILTabAssy
                            
                        oLTabAssy.Initialize oLTab, oLTabNext
                        
                        GetTabAssList.AddtoList oLTabAssy
                        
                    Else
                    
                        Set GetTabAssList = New IArrListObject
                        Exit For
                        
                    End If
                    
                ElseIf Not (oLTab.IsLeft) And Not (oLTabNext.IsLeft) Then
                
                    Dim TempTabAssy As ILTabAssy
                    Set TempTabAssy = GetSuitableAssyFromAssyList(oLTab, oLTabNext, GetTabAssList)
                    
                    If TempTabAssy Is Nothing Then
                    
                        Set GetTabAssList = New IArrListObject
                        Exit For
                    
                    Else
                        
                        TempTabAssy.AddToTabsList oLTab, oLTabNext
                        
                    End If
                        
                        
                Else
                    
                    Set GetTabAssList = New IArrListObject
                    Exit For

                End If

            Next j
            
        Else
        
            Set GetTabAssList = New IArrListObject
            Exit For
            
        End If
    
    Next i

End Function

Function GetSuitableAssyFromAssyList(oLTab As ILTabs, oLTabNext As ILTabs, ArrList As IArrListObject) As ILTabAssy

    Dim i As Integer
    
    Dim vTabAssyItems As Variant
    vTabAssyItems = ArrList.Items
    
    For i = UBound(vTabAssyItems) To LBound(vTabAssyItems) Step -1
    
        Dim oTabAssy As ILTabAssy
        Set oTabAssy = vTabAssyItems(i)
        
        If Abs(oTabAssy.LowerLeftLTab.yPoint - oLTab.yPoint) < 0.001 And _
                    Abs(oTabAssy.UpperLeftLTab.yPoint - oLTabNext.yPoint) <= 0.001 Then
                    
            Set GetSuitableAssyFromAssyList = oTabAssy
            Exit For
                    
        End If
        
    Next i
    
    
End Function
Function GetLTabList(InternalCompList As IArrListObject, swView As SldWorks.View) As IArrListObject
    
    Set GetLTabList = New IArrListObject
    
    Dim vIntComps As Variant
    vIntComps = InternalCompList.Items
    
    If Not IsEmpty(vIntComps) Then
    
        Dim i As Integer
        For i = LBound(vIntComps) To UBound(vIntComps)
        
            Dim oIntComp As IComp
            Set oIntComp = vIntComps(i)
            
            Dim vFaces As Variant
            vFaces = swView.GetVisibleEntities2(oIntComp.GetComponent, swViewEntityType_e.swViewEntityType_Face)
            
            If Not IsEmpty(vFaces) Then
                
                Dim swFace As SldWorks.Face2
            
                If UBound(vFaces) = 0 Then
                
                   Set swFace = vFaces(0)
                   
                Else
                
                    Set swFace = GetLargestFace(vFaces)
                    
                End If
                
                Dim vLoops As Variant
                vLoops = swFace.GetLoops
                
                Call AddNonHoleLoopsToList(vLoops, GetLTabList, oIntComp, swView)

            End If

        Next i
        
    End If
    
End Function

Sub AddNonHoleLoopsToList(vLoops As Variant, ArrList As IArrListObject, oComp As IComp, swView As SldWorks.View)

    Dim i As Integer
    For i = LBound(vLoops) To UBound(vLoops)
    
        Dim swLoop As SldWorks.Loop2
        Set swLoop = vLoops(i)
        
        If Not (swLoop.IsOuter) Then
        
            Dim vEdges As Variant
            vEdges = swLoop.GetEdges
            
            If UBound(vEdges) = 5 Then
            
                If Not (IsContainsCircularEdge(vEdges)) Then
                    
                    Dim oLTab As ILTabs
                    Set oLTab = New ILTabs

                    oLTab.Initialize swLoop, oComp.GetComponent, swView

                    ArrList.AddtoList oLTab
                
                End If
            
            End If
        
        
        
        End If
    
    Next i
    
    ArrList.SortItems "yPoint", False
    ArrList.SortItems "xPoint", False
    
End Sub

Function IsContainsCircularEdge(vEdges As Variant) As Boolean

    IsContainsCircularEdge = False
    
    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
    
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
        
        Dim swCurve As SldWorks.Curve
        Set swCurve = swEdge.GetCurve
        
        If swCurve.IsCircle Then
        
            IsContainsCircularEdge = True
            Exit For
            
        End If
    
    Next i
    
End Function
 
Function GetLargestFace(vFaces As Variant) As SldWorks.Face2

    Dim i As Integer
    Dim Area As Double
    Area = 0
    For i = LBound(vFaces) To UBound(vFaces)
    
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)
        
        If swFace.GetArea > Area Then
        
            Set GetLargestFace = swFace
            Area = swFace.GetArea
            
        End If

    Next i
   
End Function

Private Sub SelectComponent(swDrawing As SldWorks.DrawingDoc, oComp As IComp, xPos As Double, _
    yPos As Double, Count As Integer, IsSelected As Boolean, swView As SldWorks.View)
    
    IsSelected = swDrawing.Extension.SelectByID2("", "FACE", xPos, yPos, _
                    0, False, -1, Nothing, 1)
    
    If IsSelected Then
    
        Dim swSelectMgr As SldWorks.SelectionMgr
        Set swSelectMgr = swDrawing.SelectionManager
        
        Dim swComp As SldWorks.DrawingComponent
        Set swComp = swSelectMgr.GetSelectedObjectsComponent4(2, -1)
        
        If Not (Right(swComp.Name, Len(swComp.Name) - InStrRev(swComp.Name, "/")) = _
            Right(oComp.GetComponent.Name2, Len(oComp.GetComponent.Name2) - InStrRev(oComp.GetComponent.Name2, "/"))) Then
            
            Dim vFaces As Variant
            vFaces = swView.GetVisibleEntities2(oComp.GetComponent, swViewEntityType_e.swViewEntityType_Face)
            
            Dim swFace As SldWorks.Face2
            Set swFace = vFaces(0)
            IsSelected = SelectEntity(swFace, False, swView)
            
        End If

    End If
    
End Sub

Private Function GetConsolidatedList(ArrList As IArrListObject, ByRef DoorList As IArrListObject) As Variant

    Dim vConsolidatedLists As Variant
    
    Dim vComps As Variant
    vComps = ArrList.Items
    
    Dim IsDoorStarted As Boolean
    IsDoorStarted = False
    
    Dim oDoorAssy As IDoorAssy
    
    Set DoorList = New IArrListObject
    
    Dim k As Integer
    For k = LBound(vComps) To UBound(vComps)
    
        Dim oComp As IComp
        Set oComp = vComps(k)

        Dim List As IConsolidatedList

        If k = LBound(vComps) Then

            Set List = New IConsolidatedList
            Set List.Comp = oComp

            ReDim vConsolidatedLists(0)
            Set vConsolidatedLists(0) = List

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
            
            Dim PrevComp As IComp
            Set PrevComp = vComps(k - 1)
            
            Debug.Print PrevComp.GetComponent.Name2
            
            If Not Abs(PrevComp.yMin - oComp.yMin) <= 0.001 Then

            
                If IsDoorStarted Then
                    
                    IsDoorStarted = False
                    Set oDoorAssy.EndComp = oComp
                    Set oDoorAssy.TopComp = PrevComp
                    
                    DoorList.AddtoList oDoorAssy

                Else
                    Set oDoorAssy = New IDoorAssy
                    IsDoorStarted = True
                    Set oDoorAssy.StartComp = PrevComp
                    
                End If

            End If

        End If

    Next k

    GetConsolidatedList = vConsolidatedLists

End Function



Sub AddInsulationMaterialNote(swInsulationComp As SldWorks.Component2, _
        swView As SldWorks.View, swDrawing As SldWorks.ModelDoc2, Clearance As Double)
    
    
    swDrawing.ActivateView swView.Name
    
    Dim MaterialName As String
    MaterialName = swInsulationComp.GetModelDoc2().MaterialIdName
    
    If MaterialName = "" Then
    
        MaterialName = "INSULATION"
        
    End If
    
    Dim StringPos As Integer
    StringPos = InStr(MaterialName, "|")
    
    If StringPos > 0 Then

        MaterialName = Right(MaterialName, Len(MaterialName) - StringPos)

    End If
    
    Dim vFaces As Variant
    vFaces = swView.GetVisibleEntities2(swInsulationComp, swViewEntityType_e.swViewEntityType_Face)

    
    Dim oFace As IFaceClass
    Set oFace = GetFaceBeforeTheRightEnd(swInsulationComp, swView, vFaces)

    Dim IsSelected As Boolean

    swDrawing.ViewZoomTo2 oFace.xMin, oFace.yMin, oFace.zMin, oFace.xMax, oFace.yMax, oFace.zMax
    swDrawing.ClearSelection2 True
    
    IsSelected = SelectFaceWithPosition(swDrawing, oFace, CheckComp:=True)
    
    swDrawing.ViewZoomTo2 0, 0, 0, 17 * 0.0254, 11 * 0.0254, 0
    
    If False = IsSelected Then

        swView.SelectEntity oFace.GetFace, False

    End If
    
    If IsSelected Then
    
        Dim swAnn As SldWorks.Annotation
        Set swAnn = AddNoteToView(swDrawing, UCase(MaterialName), ((oFace.xMin + oFace.xMax) / 2) - Len(MaterialName) * 0.002, (oFace.yMin + oFace.yMax) / 2 + Clearance)
        
        swAnn.SetLeader3 swLeaderStyle_e.swBENT, swLeaderSide_e.swLS_SMART, False, False, True, False
        
        Dim HeadStyle As Integer
        HeadStyle = swAnn.SetArrowHeadStyleAtIndex(0, swArrowStyle_e.swCLOSED_ARROWHEAD)
        
    End If


End Sub

Private Function SelectFaceWithPosition(swDrawing As SldWorks.DrawingDoc, oFace As IFaceClass, _
        Optional Append As Boolean = False, Optional CheckComp As Boolean = False) As Boolean

    SelectFaceWithPosition = swDrawing.Extension.SelectByID2("", "FACE", (oFace.xMin + oFace.xMax) / 2, (oFace.yMin + oFace.yMax) / 2, _
                    0, Append, -1, Nothing, 1)
                    
    If SelectFaceWithPosition Then

        Dim swSelectMgr As SldWorks.SelectionMgr
        Set swSelectMgr = swDrawing.SelectionManager

        Dim swCompCheck As SldWorks.DrawingComponent
        Set swCompCheck = swSelectMgr.GetSelectedObjectsComponent4(2, -1)
        
        Dim swCompFace As SldWorks.Face2
        Set swCompFace = swSelectMgr.GetSelectedObject6(2, -1)
        
        If CheckComp Then

            If Not (Right(swCompCheck.Name, Len(swCompCheck.Name) - InStrRev(swCompCheck.Name, "/")) = _
                    Right(oFace.GetComponent.Name2, Len(oFace.GetComponent.Name2) - InStrRev(oFace.GetComponent.Name2, "/"))) Then

                SelectFaceWithPosition = False
                swDrawing.ClearSelection2 True

            End If

        End If
        
    End If

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
                    
            swSketchHatch.Pattern = "ISO (Steel)"
            swSketchHatch.Scale2 = swView.ScaleDecimal * 4
                
        Next i
                
    End If

End Sub

Function GetFaceBeforeTheRightEnd(swComp As SldWorks.Component2, swView As SldWorks.View, vFaces As Variant)

    If Not IsEmpty(vFaces) Then
    
        Dim FaceArrList As IArrListObject
        Set FaceArrList = New IArrListObject
        
        Dim i As Integer
        For i = LBound(vFaces) To UBound(vFaces)
        
            Dim swFace As SldWorks.Face2
            Set swFace = vFaces(i)
            
            Dim oFace As IFaceClass
            Set oFace = New IFaceClass
            oFace.Initialize swFace, swView, swComp
            
            FaceArrList.AddtoList oFace
            
        Next i
        
        FaceArrList.SortItems "xMin"
        vFaces = FaceArrList.Items
        
        If UBound(vFaces) >= 1 Then
        
            Set GetFaceBeforeTheRightEnd = vFaces(1)
            
        Else
        
            Set GetFaceBeforeTheRightEnd = vFaces(0)
            
        End If
        
    End If

End Function

Private Sub AddInsulationHatches(swInsulationComp As SldWorks.Component2, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc)
    
    swDrawing.ClearSelection2 True
    
    Dim vFaces As Variant
    vFaces = swView.GetVisibleEntities2(swInsulationComp, swViewEntityType_e.swViewEntityType_Face)
    
    If Not IsEmpty(vFaces) Then
    
        Dim i As Integer
        For i = LBound(vFaces) To UBound(vFaces)
    
            Dim swFace As SldWorks.Face2
            Set swFace = vFaces(i)
            
            swView.SelectEntity swFace, True
            
        Next i
        
        swDrawing.InsertHatchedFace
        Call UpdateHatchProperties(swView)
        
    End If

End Sub


Private Sub UpdateMaxMinPoints(vComps As Variant, swView As SldWorks.View)

    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim oComp As IComp
        Set oComp = vComps(i)

        Dim MinPoint As Variant
        MinPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
        Dim MaxPoint As Variant
        MaxPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
        
        oComp.UpdateSheetMaxMinDimensions swView, MinPoint, MaxPoint
        
    Next i

End Sub
 
Private Sub AddCastingSketchAndNote(oComp As IComp, swView As SldWorks.View, swSketchMgr As SldWorks.SketchManager, _
                swDrawing As SldWorks.DrawingDoc)
    
    swDrawing.ActivateView swView.Name
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
        
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
   
    Dim swSketchSegment As SldWorks.SketchSegment
    Set swSketchSegment = swSketchMgr.CreateLine(xMax - 0.25 * 0.0254, yMin, _
                                0, xMax + 16 * 0.0254, yMin, 0)
                                
    swSketchSegment.ConstructionGeometry = True
        
    Dim vSketchPoint As Variant
    vSketchPoint = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, True, 0.5)
    Call AddNoteToView(swDrawing, "CASTING BED", vSketchPoint(0) - 0.02, vSketchPoint(1) - 0.01)
        
    Dim swEdge As SldWorks.Edge
    Set swEdge = GetEdgeInView(oComp, swView, True, False)
        
    Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)
    
End Sub

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
        
        Dim swlayer As SldWorks.Layer
        Set swlayer = swLayerMgr.GetLayer(LayName)
        
        swlayer.Style = swLineStyles_e.swLineCENTER
        swlayer.Width = swLineWeights_e.swLW_THICK5
        
    End If
    
End Sub

Private Sub SketchLineForNonCornerPanels(swView As SldWorks.View, wallName As String, _
        swDrawing As SldWorks.ModelDoc2, oSubAssy As ISubAssy, swBottomEdge As SldWorks.Edge, _
            ByRef MaxClearance As Double, ByRef swStartSketch As SldWorks.SketchSegment, ByRef swEndSketch As SldWorks.SketchSegment)
    
    swDrawing.ActivateView swView.Name
    
    If InStr(wallName, "Wall") > 0 And (InStr(wallName, "-A") > 0 Or InStr(wallName, "-B") Or _
                                InStr(wallName, "-C") Or InStr(wallName, "-D")) Then
    
        Dim viewDrawComp As SldWorks.DrawingComponent
        Set viewDrawComp = swView.RootDrawingComponent
        
        Dim viewComp As SldWorks.Component2
        Set viewComp = viewDrawComp.Component
        
        Debug.Print viewComp.Name2
    
        Dim swControlSketch As SldWorks.Component2
        Set swControlSketch = GetControlSketch
        
        Dim PlaneName As String
        
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
        
        
        

    End If
    

End Sub

Private Sub AddDimensionFromCornerSketches(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
                swStartSketch As SldWorks.SketchSegment, swEndSketch As SldWorks.SketchSegment, _
            oSubAssy As ISubAssy, ByRef MaxClearance As Double)

    Dim swDisplayDim As SldWorks.DisplayDimension
        
        If Not swStartSketch Is Nothing Then
        
            MaxClearance = MaxClearance + 0.006
            
            If Not swEndSketch Is Nothing Then
                
                swStartSketch.Select4 False, Nothing
                swEndSketch.Select4 True, Nothing

            Else
            
                swStartSketch.Select4 False, Nothing
                Call SelectEntity(oSubAssy.EndEdge, True, swView)
    
            End If
            
        Else

            If Not swEndSketch Is Nothing Then
            
                MaxClearance = MaxClearance + 0.006
                swEndSketch.Select4 False, Nothing
                Call SelectEntity(oSubAssy.StartEdge, True, swView)
            
            End If
            
        End If
        
        Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.StartComp.xMin + 0.01, oSubAssy.EndComp.yMin - MaxClearance, 0)
        If Not swDisplayDim Is Nothing Then

            swDisplayDim.CenterText = True
            swDisplayDim.SetDual2 False, False
            
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


Private Sub AddOverallDimension(oSubAssy As ISubAssy, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, ByRef Clearance As Double)
    
    Clearance = Clearance + 0.006
    
    Dim swDisplayDim As SldWorks.DisplayDimension
    Set swDisplayDim = SelectAndAddDimension(oSubAssy.StartEdge, oSubAssy.EndEdge, swDrawing, _
                oSubAssy.EndComp.xMin - 0.01, oSubAssy.EndComp.yMin - Clearance, swView)
    Set oSubAssy.Dimension = swDisplayDim
    
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
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView, IsParanthesis:=True)
                                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(subAssylist.Items(UBound(subAssylist.Items)).EndEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView, IsParanthesis:=True)
                                
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
                            oComp.xMax + 0.01, vSheetPoint(1) - 0.005, swView, IsParanthesis:=True)
                            
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

Private Sub AddSplitLineNote(swSketchSegment As SldWorks.SketchLine, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            NoteText As String, Optional IsRight As Boolean = True, Optional ClearanceVal As Double = 0.005)

    
    Dim vPointInSheet As Variant
    vPointInSheet = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, False)

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
        
        TempDict.add FlatCompList(i).GetComponent.Name2, FlatCompList(i)
        CompNoDict.add FlatCompList(i).GetComponent.Name2, i
    
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
                
                compDict.add swComp.Name2, swComp
            
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

Private Function AddDimensionInFrontView(swView As SldWorks.View, FlatCompList As Variant, _
            MaxCompHeight As IComp, swDrawing As SldWorks.ModelDoc2, _
            ByRef swLeftEdge As SldWorks.Edge, ByRef swRightEdge As SldWorks.Edge) As SldWorks.Edge
            
    If Not IsEmpty(FlatCompList) Then
            
        Dim vOutline As Variant
        vOutline = swView.GetOutline
    
        Dim LeftComp As IComp
        Set LeftComp = FlatCompList(0)
        
        Dim RightComp As IComp
        Set RightComp = FlatCompList(UBound(FlatCompList))
        
        Dim ClearanceLeft As Double
        Dim ClearanceRight As Double
        
        ClearanceLeft = GetClearance(LeftComp)
        ClearanceRight = GetClearance(RightComp)
    
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
                            swBottomRightEdge, swDrawing, RightComp.xMax + ClearanceRight, (vOutline(1) + vOutline(3)) / 2, swView, IsHorizontalDim:=False)
        Else
        
            Dim swBottomLeftEdge As SldWorks.Edge
            Set swBottomLeftEdge = GetEdgeInView(LeftComp, swView, True, False)
            
            Dim swTopLeftEdge As SldWorks.Edge
            Set swTopLeftEdge = GetEdgeInView(LeftComp, swView, True, True)
            
            Set swRightDim = SelectAndAddDimension(swTopRightEdge, _
                            swBottomRightEdge, swDrawing, RightComp.xMax + ClearanceRight, (vOutline(1) + vOutline(3)) / 2, swView, IsHorizontalDim:=False)
                            
            Dim swLeftDim As SldWorks.DisplayDimension
            Set swLeftDim = SelectAndAddDimension(swTopLeftEdge, _
                swBottomLeftEdge, swDrawing, LeftComp.xMin - ClearanceLeft, (vOutline(1) + vOutline(3)) / 2, swView, IsHorizontalDim:=False)
            
        End If
        
        Set AddDimensionInFrontView = swBottomRightEdge
        
    End If

End Function

Private Function GetClearance(oComp As IComp) As Double

    If InStr(oComp.GetCustomProperty("Profile"), "CORNER") > 0 Then
        
        GetClearance = 0.01
        
    Else
        
        GetClearance = 0.02
        
    End If
        
End Function

Private Function SelectAndAddDimension(swEdge1 As SldWorks.Edge, swEdge2 As SldWorks.Edge, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, yPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True, _
                Optional IsParanthesis As Boolean = False, Optional IsHorizontalDim As Boolean = True) As SldWorks.DisplayDimension
    
    If Not (swEdge1 Is Nothing) And Not (swEdge2 Is Nothing) Then
        
        swDrawing.ClearSelection2 True
        
        swView.FocusLocked = True
        
        Call SelectEntity(swEdge1, False, swView)
        Call SelectEntity(swEdge2, True, swView)
        
        If IsHorizontalDim Then
        
            Set SelectAndAddDimension = swDrawing.AddHorizontalDimension2(xPos, yPos, 0)
            
            If SelectAndAddDimension Is Nothing Then
                
                Set SelectAndAddDimension = swDrawing.AddVerticalDimension2(xPos, yPos, 0)
                
            End If
            
        Else
            
            Set SelectAndAddDimension = swDrawing.AddVerticalDimension2(xPos, yPos, 0)
            
            If SelectAndAddDimension Is Nothing Then
                
                Set SelectAndAddDimension = swDrawing.AddHorizontalDimension2(xPos, yPos, 0)
                
            End If
            
        End If

        If Not SelectAndAddDimension Is Nothing Then
        
            SelectAndAddDimension.CenterText = True
            
            If IsDual Then
            
                SelectAndAddDimension.SetDual2 False, False
                
            End If
            
            If IsParanthesis Then
            
                SelectAndAddDimension.ShowParenthesis = True
                
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
    
    Call StrucutralInternal.GetMaxMinPoint(vViewMinPt(0), vViewMaxPt(0), xMin, xMax)
    Call StrucutralInternal.GetMaxMinPoint(vViewMinPt(1), vViewMaxPt(1), yMin, yMax)
    
End Sub
 
'Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, Is12GAPanelExists As Boolean, _
'            IsAllPanels12GA As Boolean, IsDoorExists As Boolean, ByRef NoteCount As Integer, wallName As String) As SldWorks.Note
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
'End Function

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

Private Sub CleanUpActivateAndAddViewLabel(swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, wallName As String, _
        yPos As Double, Optional xPos As Double = 0, Optional InsulationName As String = "")

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

        SheetDesc = "STRUCTURAL, ELEVATION, INTERNAL LINER PANELS, " & UCase(wallName)


    Else
        
        SheetDesc = "STRUCTURAL, " & UCase(wallName) & ", INTERNAL LINER PANELS"
        
    End If
    
    LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34)
    swDrawing.Extension.CustomPropertyManager("").Set2 "SHEET DESCRIPTION", SheetDesc
    swDrawing.Extension.CustomPropertyManager("").Set2 "ISSUED FOR", "CONSTRUCTION"
    
    If xPos = 0 Then
    
        Dim vOutline As Variant
        vOutline = swView.GetOutline
        xPos = (vOutline(0) + vOutline(2)) / 2
        
    End If
    
    
    Dim swLabelNote As SldWorks.Note

    Set swLabelNote = swDrawing.CreateText2(LabelText, xPos, yPos, 0, 0, 0)
    swLabelNote.SetTextJustification swTextJustification_e.swTextJustificationCenter
    
    swDrawing.Extension.Rebuild swRebuildOptions_e.swCurrentSheetDisp

End Sub


Private Sub AddCollinearRelation(swDrawing As SldWorks.DrawingDoc, swEdge As SldWorks.Edge, swSketchSegment As SldWorks.SketchSegment, swView As SldWorks.View)
    
    If Not (swEdge Is Nothing) And Not (swSketchSegment Is Nothing) Then
    
        Call SelectEntity(swEdge, False, swView)
        swSketchSegment.Select4 True, Nothing
                
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
    End If
    
End Sub

Function GetEdgeInView(oComp As IComp, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
    
    Dim Idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        Idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        Idx = 0
        
        If IsMax Then
        
            ValToMatch = xMax
            
        Else
        
             ValToMatch = xMin
             
        End If
        
    End If
    
    Dim swComp As SldWorks.Component2
    Set swComp = oComp.GetComponent
    

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
        
    Else
    
        vEnts = GetComponentEdges(swComp)
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)
            
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
                
                If Abs(vStartPoint(Idx) - vEndPoint(Idx)) <= 0.00001 And Abs(vStartPoint(Idx) - ValToMatch) <= 0.00001 Then
                    
                    Dim vCurveParam As Variant
                    vCurveParam = swEdge.GetCurveParams2
                    
                    If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                        
                        TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                        Set GetEdgeInView = swEdge
                        
                    End If
                    
                End If
            
            End If
            
        Next i

    End If

End Function


Function AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, yPos As Double) As SldWorks.Annotation
            
    Dim swNote As SldWorks.Note
    Set swNote = swDrawing.InsertNote(NoteText)
    
    Dim swAnnotation As SldWorks.Annotation
            
    If Not swNote Is Nothing Then

        Set swAnnotation = swNote.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, yPos, 0

        End If

    End If
    
    Set AddNoteToView = swAnnotation
    
End Function

Function SelectSketchSegment(swSketchSegment As SldWorks.SketchSegment, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, Append As Boolean, Optional IsNearEnd As Boolean = True, Optional PercentFromEnd As Double = 0.01)
    
    Dim swSketchLine As SldWorks.SketchLine
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
    vPointInSheet = StrucutralInternal.GetSketchPointInSheetSpace(swView, vSketchPoint)
    
    swDrawing.Extension.SelectByID2 "Line" & swSketchSegment.GetID(1), "SKETCHSEGMENT", vPointInSheet(0), vPointInSheet(1), vPointInSheet(2), Append, -1, Nothing, 0
    SelectSketchSegment = vPointInSheet
    
End Function

Private Sub AddCallouts(vConsolidatedList As Variant, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, MaxCompHeight As Double)
    
    Const SheetPosForLastBalloon As Double = 0.266
    Const Increment As Double = 0.005
    Const MaxBalloonWidth As Double = 0.015875
    
    swDrawing.Extension.SetUserPreferenceInteger swUserPreferenceIntegerValue_e.swDetailingBOMUpperText, swUserPreferenceOption_e.swDetailingNoOptionSpecified, swBalloonTextContent_e.swBalloonTextPartNumberBOM
    
    Dim maxNoOfBalloons As Integer
    maxNoOfBalloons = 2 'Int((SheetPosForLastBalloon - MaxCompHeight) / Increment)
    
    Dim AddorSub As Integer
    Dim BalloonCount As Integer
    
    AddorSub = 1
    BalloonCount = 1
    
    Dim annXPos As Double
    Dim annYPos As Double
    
    Dim i As Integer
    For i = LBound(vConsolidatedList) To UBound(vConsolidatedList)
    
        Dim oList As IConsolidatedList
        Set oList = vConsolidatedList(i)
        
        Dim oComp As IComp
        Set oComp = oList.Comp

        swDrawing.ClearSelection2 True

        Dim xPos As Double
        Dim yPos As Double
      
        xPos = (oComp.xMin + oComp.xMax) / 2 '(oComp.xMin + oComp.xMax) / 2 - Abs((oComp.xMin - oComp.xMax) / 2) + 3.5 * 0.0254 * swView.ScaleDecimal
        yPos = 0.075 * oComp.yMin + 0.925 * oComp.yMax

        If Not (i = LBound(vConsolidatedList)) Then
    
            Dim PrevComp As IComp
            Set PrevComp = vConsolidatedList(i - 1).Comp
    
            If AddorSub = -1 Then
    
                If Abs((PrevComp.xMin + PrevComp.xMax) / 2 - (oComp.xMin + oComp.xMax) / 2) > 2 * MaxBalloonWidth Or _
                    Abs((PrevComp.xMin + PrevComp.xMax) / 2 - (oComp.xMin + oComp.xMax) / 2) > MaxBalloonWidth And BalloonCount > 2 Then
    
                        AddorSub = 1
                        BalloonCount = 1
    
                End If
    
            Else
    
                If Abs((PrevComp.xMin + PrevComp.xMax) / 2 - (oComp.xMin + oComp.xMax) / 2) > MaxBalloonWidth Then
    
                    AddorSub = 1
                    BalloonCount = 1
    
                End If
    
            End If

        End If
            
        If AddorSub = 1 Then
            
            If BalloonCount > maxNoOfBalloons Then
                    
                AddorSub = -1
                BalloonCount = BalloonCount + AddorSub
                    
            End If
            
        Else
            
            If BalloonCount < 1 Then

                BalloonCount = maxNoOfBalloons

            End If
                
        End If
            
        annXPos = xPos
        annYPos = MaxCompHeight + BalloonCount * Increment
        BalloonCount = BalloonCount + AddorSub
 

        Dim IsSelected As Boolean
        IsSelected = False
        Call SelectComponent(swDrawing, oComp, xPos, yPos, 1, IsSelected, swView)
        
        If IsSelected Then
        
            Dim swBalloonParams As SldWorks.BalloonOptions
            Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
            swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
            swBalloonParams.Style = swBalloonStyle_e.swBS_Inspection
            
            If oList.Qty > 1 Then
    
                swBalloonParams.ShowQuantity = True
                swBalloonParams.QuantityOverride = True
                swBalloonParams.QuantityOverrideValue = CStr(oList.Qty)
                
            End If
            
            Dim swComp As SldWorks.Component2
            Set swComp = oComp.GetComponent
            'Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))
            
            Dim swNote As SldWorks.Note
            Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
            
            If Not swNote Is Nothing Then

                Dim swAnn As SldWorks.Annotation
                Set swAnn = swNote.GetAnnotation
                swAnn.SetPosition2 annXPos, annYPos, 0
                
                Dim HeadStyle As Integer
                
                swAnn.SetLeader3 swLeaderStyle_e.swAlwaysAttachToBalloon + swLeaderStyle_e.swSTRAIGHT, swLeaderSide_e.swLS_SMART, False, False, True, False
                HeadStyle = swAnn.SetArrowHeadStyleAtIndex(0, swArrowStyle_e.swCLOSED_ARROWHEAD)

                If AddorSub = 1 Then
                    
                    Dim vNoteExtents As Variant
                    vNoteExtents = swNote.GetExtent
     
                    If oList.Qty > 1 Then
                        
                        annXPos = xPos - ((vNoteExtents(3) - vNoteExtents(0))) + 0.0064
                        
                    Else
                        annXPos = xPos - ((vNoteExtents(3) - vNoteExtents(0))) + 0.0027
                            
                    End If
            
                    swAnn.SetPosition2 annXPos, annYPos, 0
    
                End If
                    

            End If
            
        End If
        
    Next i

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

Function ScaleView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
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

Function GetComponentsSortedWithXPosition(swView As SldWorks.View, swDrawing As SldWorks.ModelDoc2, _
       ProfileTextToMatch As String) As IArrListObject
    
    swDrawing.ActivateView swView.Name

    Dim vComps As Variant
    vComps = swView.GetVisibleComponents()

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
            
            Dim swBody As SldWorks.Body2
            Set swBody = swCompFromRoot.GetBody
            
            If Not swBody Is Nothing Then
                
                If swBody.IsSheetMetal Then
            
                    Dim Profile As String
                    Dim ResolvedVal As String
                    Dim wasResolved As Boolean
                    swCompProp.Get5 "Profile", False, Profile, ResolvedVal, wasResolved
                    
                    If InStr(Profile, ProfileTextToMatch) > 0 Then
                    
                        CompList.AddtoList GetComponentWithPosition(swCompFromRoot, swView, swDrawing)
                    
                    End If
                    
                End If
                
            End If
        
        End If
        
    Next i

    CompList.SortItems "xMin", False
        
    Set GetComponentsSortedWithXPosition = CompList

End Function

Sub GetComponentBoundsInView(CompList As IArrListObject, ByRef ViewWidth As Double, _
    ByRef ViewHeight As Double, MaxHeightComp As IComp)

    CompList.SortItems "yMin", False
    
    Dim MinHeight As Double
    MinHeight = CompList.Items(LBound(CompList.Items)).yMin
    
    CompList.SortItems "yMax"
    Set MaxHeightComp = CompList.Items(LBound(CompList.Items))
    
    ViewHeight = MaxHeightComp.yMax - MinHeight
    
    CompList.SortItems "yMax", True
    CompList.SortItems "xMin", False
    ViewWidth = CompList.Items(UBound(CompList.Items)).xMax - CompList.Items(LBound(CompList.Items)).xMin

    
End Sub

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
        swDrawing As SldWorks.ModelDoc2) As IComp

    Dim MinPoint As Variant
    Dim MaxPoint As Variant
    Dim vBodyMinPoint(2) As Double
    Dim vBodyMaxPoint(2) As Double
    Call GetMinMaxBodyPointsInSheetSpace(swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, swView)
            
    Dim oComp As IComp
    Set oComp = New IComp
    oComp.Initialize swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint

    Set GetComponentWithPosition = oComp

End Function

Private Sub GetMinMaxBodyPointsInSheetSpace(swComp As SldWorks.Component2, _
        ByRef MinPoint As Variant, ByRef MaxPoint As Variant, ByRef vBodyMinPoint() As Double, _
            ByRef vBodyMaxPoint() As Double, swView As SldWorks.View, Optional IsCorZ As Boolean = False)
            
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

            
    MinPoint = GetComponentPointInSheetSpace(swComp, vBodyMinPoint, swView)
    MaxPoint = GetComponentPointInSheetSpace(swComp, vBodyMaxPoint, swView)
    
End Sub

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

