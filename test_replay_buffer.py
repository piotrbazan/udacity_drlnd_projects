from replay_buffer import ReplayBuffer


def test_reply_buffer_store():
    memory = ReplayBuffer(2)
    assert len(memory) == 0
    memory.store(1, 2, 3, 4, False)
    assert len(memory) == 1
    memory.store(2, 3, 4, 5, False)
    assert len(memory) == 2
    memory.store(3, 4, 5, 6, True)
    assert len(memory) == 2
    assert memory.dones[0]
    assert not memory.dones[1]


def test_reply_buffer_sample():
    memory = ReplayBuffer(10)
    for i in range(10):
        memory.store([1, 2, 3], 2, 3, [4, 5, 6], False)

    s, a, r, ns, d = memory.sample(2)
    assert s.shape == (2, 3)
    assert a.shape == (2, 1)
    assert r.shape == (2, 1)
    assert ns.shape == (2, 3)
    assert d.shape == (2, 1)


def test_state_dict():
    memory = ReplayBuffer(2)
    assert memory.state_dict() == dict(size=0)
    memory.store(1, 2, 3, 4, 5)
    assert memory.state_dict() == dict(size=1)
    memory.store(1, 2, 3, 4, 5)
    assert memory.state_dict() == dict(size=2)
