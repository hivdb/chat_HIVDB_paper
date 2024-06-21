SELECT
    DISTINCT Species
FROM
    _InVitroIso
WHERE
    IsolateID IN (
        SELECT IsolateID
        FROM tblInVitroSelection
        WHERE RefID IN (
            SELECT RefID
            FROM tblReferences
            WHERE MedlineID = {pubmed_id}
        )
    )
;
