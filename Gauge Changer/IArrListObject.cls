VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IArrListObject"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private vItems As Variant
Private IsInit As Boolean

Property Get Items() As Variant
    
    Items = vItems

End Property

Private Sub Class_Initialize()

    IsInit = True

End Sub

Function AddtoList(Val As Object)

    If IsInit Then
        
        IsInit = False
        ReDim vItems(0)
        Set vItems(0) = Val
    
    Else
    
        ReDim Preserve vItems(UBound(vItems) + 1)
        Set vItems(UBound(vItems)) = Val
    
    End If
    
End Function

Public Sub SortItems(SortParam As String)

    vItems = MergeSort(vItems, SortParam)

End Sub

Private Function MergeSort(vItems As Variant, SortParam As String)

    If UBound(vItems) = 0 Then
        
        MergeSort = vItems
    
    Else
     
        Dim Mid As Integer
        Mid = Int((UBound(vItems) - LBound(vItems) + 1) / 2)
        
        Dim LeftArr() As Variant
        Dim RightArr() As Variant
        
        Dim i As Integer
        For i = LBound(vItems) To Mid - 1
        
            ReDim Preserve LeftArr(i)
            LeftArr(UBound(LeftArr)) = vItems(i)
        
        Next i
        
        Dim j As Integer
        For j = Mid To UBound(vItems)
        
            ReDim Preserve RightArr(j - Mid)
            RightArr(UBound(RightArr)) = vItems(j)
        
        Next j
        
        LeftArr = MergeSort(LeftArr)
        RightArr = MergeSort(RightArr)
        
        MergeSort = MergeArray(vItems, RightArr, LeftArr)
        
    End If

End Function

Private Function MergeArray(vItems As Variant, RightArr As Variant, LeftArr As Variant, SortParam As String) As Variant

    Dim i As Integer
    Dim j As Integer
    Dim k As Integer
    
    i = 0
    j = 0
    k = 0
    
    While (i <= UBound(LeftArr) And j <= UBound(RightArr))
    
        If CallByName(LeftArr(i), SortParam, VbGet) >= CallByName(RightArr(j), SortParam, VbGet) Then
        
            vItems(k) = LeftArr(i)
            k = k + 1
            i = i + 1
                
        Else
            
            vItems(k) = RightArr(j)
            k = k + 1
            j = j + 1

        End If
    
    Wend
    
    While (i <= UBound(LeftArr))
    
        vItems(k) = LeftArr(i)
        k = k + 1
        i = i + 1
        
    Wend
    
    While (j <= UBound(RightArr))
    
        vItems(k) = RightArr(j)
        k = k + 1
        j = j + 1
        
    Wend
    
    MergeArray = vItems

End Function

