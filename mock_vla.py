# mock vla model to test object reasoning 

class MockVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = torch.nn.Linear(512 * 5, 7)  # Mock fusion
    
    def forward(self, tokens, objects):
        # Object-centric reasoning
        obj_features = torch.mean(torch.stack([t['embedding'] for t in objects]), dim=0)
        fused = torch.cat([tokens.mean(0), obj_features])
        actions = self.action_head(fused)
        return actions

model = MockVLA()