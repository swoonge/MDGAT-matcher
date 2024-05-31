def test_forward_FPFH():
    model = YourModel()  # Instantiate your model here
    data = {
        'keypoints0': torch.tensor([...]),  # Provide sample keypoints data
        'keypoints1': torch.tensor([...]),
        'descriptors0': torch.tensor([...]),  # Provide sample descriptors data
        'descriptors1': torch.tensor([...]),
        'cloud0': torch.tensor([...]),  # Provide sample cloud data
        'cloud1': torch.tensor([...]),
        'scores0': torch.tensor([...]),  # Provide sample scores data
        'scores1': torch.tensor([...]),
        'gt_matches0': torch.tensor([...]),  # Provide sample ground truth matches data
        'gt_matches1': torch.tensor([...])
    }
    output = model.forward(data)
    # Add assertions to check the output
    assert 'matches0' in output
    assert 'matches1' in output
    assert 'matching_scores0' in output
    assert 'matching_scores1' in output
    assert 'skip_train' in output