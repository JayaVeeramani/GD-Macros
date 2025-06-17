VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IWeldBody"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cBody As SldWorks.Body2
Private cComp As SldWorks.Component2
Private cxMin As Double
Private cyMin As Double
Private cZMin As Double

Private cxMax As Double
Private cyMax As Double
Private cZMax As Double

Private cView As SldWorks.View
Private cBodyMinPoint(2) As Double
Private cBodyMaxPoint(2) As Double
Private cIsVertical As Boolean
Private cCutlist As ICutlist

Private cBeforeSubWeldment As IArrListObject
Private IsBeforeWeldmentSorted As Boolean

Private cAfterSubWeldment As IArrListObject
Private IsAfterWeldmentSorted As Boolean
Private IsSubWeldmentChecked As Boolean
Private cIsSubWeldment As Boolean

Private cAfterCPlateList As IArrListObject
Private cBeforeCPlateList As IArrListObject
Dim IsAfterCPlatesSorted As Boolean
Dim IsBeforeCPlatesSorted As Boolean

Public IsPerimeter As Boolean
Public BeforeBody As IWeldBody
Public AfterBody As IWeldBody

Private orgViewPosition As Variant
Private orgScaleDecimal As Double


Property Get Length() As Double

    Length = cCutlist.Length
    
End Property

Property Get AfterSubWeldments() As IArrListObject

    If False = IsAfterWeldmentSorted Then
    
        If cIsVertical Then
        
            cAfterSubWeldment.SortItems "yMin", False
            
        Else
        
            cAfterSubWeldment.SortItems "xMin", False
            
        End If
        IsAfterWeldmentSorted = True
        
    End If
    
     Set AfterSubWeldments = cAfterSubWeldment
    
End Property

Property Get IsSubWeldment() As Boolean
    
    If False = IsSubWeldmentChecked Then

        If CheckWhetherIsSubweldment(cAfterSubWeldment) Then
            
            cIsSubWeldment = True
        
        ElseIf CheckWhetherIsSubweldment(cBeforeSubWeldment) Then
        
            cIsSubWeldment = True
            
        End If
        
        IsSubWeldmentChecked = True
        
    End If
    
    IsSubWeldment = cIsSubWeldment
End Property

Private Function CheckWhetherIsSubweldment(ArrList As IArrListObject) As Boolean

    CheckWhetherIsSubweldment = False
    
    If ArrList.Count > 0 Then
    
        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            Dim Desc As String
            Desc = oWeldBody.Cutlist.Description
            
            If Left(Desc, 1) = "W" Or Left(Desc, 1) = "C" Or Left(Desc, 2) = "MC" Then
            
                CheckWhetherIsSubweldment = True
                Exit For
            
            End If
        
        Next i
        
    End If

End Function

Property Get AfterConnectingPlates() As Variant
    
    If False = IsAfterCPlatesSorted Then
    
        If cIsVertical Then
        
            cAfterCPlateList.SortItems "yMin", False
            
        Else
        
            cAfterCPlateList.SortItems "xMin", False
            
        End If
        
        IsAfterCPlatesSorted = True
        
    End If


    AfterConnectingPlates = cAfterCPlateList.Items
    
    
End Property

Property Get BeforeConnectingPlates() As Variant

    If False = IsBeforeCPlatesSorted Then
    
        If cIsVertical Then
        
            cBeforeCPlateList.SortItems "yMin", False
            
        Else
        
            cBeforeCPlateList.SortItems "xMin", False
            
        End If
        
        IsBeforeCPlatesSorted = True
        
    End If


    BeforeConnectingPlates = cBeforeCPlateList.Items
    
    
End Property


Property Get BeforeSubWeldments() As IArrListObject

    If False = IsBeforeWeldmentSorted Then

        If cIsVertical Then
        
            cBeforeSubWeldment.SortItems "yMin", False
            
        Else
        
            cBeforeSubWeldment.SortItems "xMin", False
            
        End If
        
        IsBeforeWeldmentSorted = True
        
    End If
    
    Set BeforeSubWeldments = cBeforeSubWeldment
    
    
End Property

Property Get IsAfterSubweldment() As Boolean
    
    IsAfterSubweldment = False
    
    If cAfterSubWeldment.Count > 0 Then
    
        IsAfterSubweldment = True
        
    End If
    
End Property

Property Get IsBeforeSubweldment() As Boolean

    IsBeforeSubweldment = False
    
    If cBeforeSubWeldment.Count > 0 Then
    
        IsBeforeSubweldment = True
        
    End If
    
End Property

Property Get GetComponent() As SldWorks.Component2

    Set GetComponent = cComp
    
End Property

Property Get Cutlist() As ICutlist
    
    Set Cutlist = cCutlist
    
End Property

Property Get IsVertical() As Boolean
    
    IsVertical = cIsVertical
    
End Property

Property Get GetBody() As SldWorks.Body2

    Set GetBody = cBody
    
End Property


Property Get xMin() As Double
 
    xMin = cxMin
    
End Property

Property Get yMin() As Double

    yMin = cyMin
    
End Property

Property Get zMin() As Double

    zMin = cZMin
    
End Property

Property Get xMax() As Double

    xMax = cxMax
    
End Property

Property Get yMax() As Double

    yMax = cyMax
    
End Property

Property Get zMax() As Double

    zMax = cZMax
    
End Property

Sub Initialize(swComp As SldWorks.Component2, swBody As SldWorks.Body2, oCutlist As ICutlist, swView As SldWorks.View)
    
    Set cComp = swComp
    Set cBody = swBody
    Set cView = swView
    Set cCutlist = oCutlist
    
    orgViewPosition = cView.Position
    orgScaleDecimal = cView.ScaleDecimal

    Dim vBodyBounds As Variant
    vBodyBounds = swBody.GetBodyBox
        
    cBodyMinPoint(0) = vBodyBounds(0)
    cBodyMinPoint(1) = vBodyBounds(1)
    cBodyMinPoint(2) = vBodyBounds(2)
        
    cBodyMaxPoint(0) = vBodyBounds(3)
    cBodyMaxPoint(1) = vBodyBounds(4)
    cBodyMaxPoint(2) = vBodyBounds(5)

    Call UpdateSheetMaxMinDimensions(swView)
    
    Set cBeforeSubWeldment = New IArrListObject
    Set cAfterSubWeldment = New IArrListObject
    
    Set cAfterCPlateList = New IArrListObject
    Set cBeforeCPlateList = New IArrListObject
    
    If Abs(cxMin - cxMax) > Abs(cyMin - cyMax) Then
        
        cIsVertical = False
            
    Else
        
        cIsVertical = True
        
    End If
    
    IsAfterWeldmentSorted = False
    IsBeforeWeldmentSorted = False
    IsSubWeldmentChecked = False
    cIsSubWeldment = False
    
    IsAfterCPlatesSorted = False
    IsBeforeCPlatesSorted = False

End Sub

Private Sub UpdateSheetMaxMinDimensions(swView As SldWorks.View)

    Dim MinPoint As Variant
    MinPoint = GetComponentPointInSheetSpace(cComp, cBodyMinPoint, swView)

    Dim MaxPoint As Variant
    MaxPoint = GetComponentPointInSheetSpace(cComp, cBodyMaxPoint, swView)
    
    Call GetMaxMinPoint(MinPoint(0), MaxPoint(0), cxMin, cxMax)
    Call GetMaxMinPoint(MinPoint(1), MaxPoint(1), cyMin, cyMax)
    Call GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)
    
End Sub

Sub CheckForUpdateInMaxMinDimensions(swView As SldWorks.View)

    If IsViewChanged(swView) Then
        
        Call UpdateSheetMaxMinDimensions(swView)
        Set cView = swView
        orgScaleDecimal = swView.ScaleDecimal
        orgViewPosition = swView.Position
            
    End If

End Sub

Private Function IsViewChanged(swView As SldWorks.View) As Boolean
    
    IsViewChanged = False

    If (swView.Name = cView.Name) Then

        Dim actualViewPosition As Variant
        actualViewPosition = swView.Position
    
        If Not (swView.ScaleDecimal = orgScaleDecimal) Or Not (orgViewPosition(0) = actualViewPosition(0)) _
                Or Not (orgViewPosition(1) = actualViewPosition(1)) Or Not (swView.Angle = cView.Angle) Then

            IsViewChanged = True
     
        End If
            
    Else
            
        IsViewChanged = True
            
    End If
    
End Function

Sub AddToSubWeldmentList(IsAfter As Boolean, oWeldBody As IWeldBody)

    If IsAfter Then
    
        cAfterSubWeldment.AddtoList oWeldBody
        
    Else
    
        cBeforeSubWeldment.AddtoList oWeldBody
        
    End If

End Sub

Sub AddToConnectingPlateList(IsAfter As Boolean, oComp As IComp)

    If IsAfter Then
    
        cAfterCPlateList.AddtoList oComp
        
    Else
    
        cBeforeCPlateList.AddtoList oComp
        
    End If

End Sub



